#!/usr/bin/env python2

import argparse
import simpy
import pandas as pd
import numpy as np
import sys, os
import random
import json

# default cmdline args
cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--config', type=str, help='JSON config file to control the simulations', required=True)

def clear_file(filename):
    with open(filename, 'w') as f:
        f.write('')

class Logger(object):
    def __init__(self, env, filename=None):
        self.env = env
        self.filename = filename

    @staticmethod
    def init_params():
        Logger.debug = NBodySimulator.config['debug'].next()

    def set_filename(self, filename):
        self.filename = os.path.join(NBodySimulator.out_run_dir, filename)
        clear_file(self.filename)

    def log(self, s):
        if Logger.debug:
            data = '{}: {}'.format(self.env.now, s)
            print data
            with open(self.filename, 'a') as f:
                f.write(data + '\n')


###################
## Message Types ##
###################

class Message(object):
    """Base Message class
    """
    count = 0
    def __init__(self, src, dst, iteration):
        self.src = src
        self.dst = dst
        self.iteration = iteration
        self.ID = Message.count
        Message.count += 1

    @staticmethod
    def init_params():
        pass

    def __str__(self):
        return "Message {}: src={}, dst={}, iteration={}".format(self.ID, self.src, self.dst, self.iteration)

class Start(Message):
    def __str__(self):
        return "(Start) " + super(Start, self).__str__()

class TraversalReq(Message):
    def __init__(self, src, dst, iteration, traversal_src, sources, expected):
        super(TraversalReq, self).__init__(src, dst, iteration)
        self.traversal_src = traversal_src
        self.sources = sources
        self.expected = expected

    def __str__(self):
        return "(TraversalReq) " + super(TraversalReq, self).__str__()

class TraversalResp(Message):
    def __init__(self, src, dst, iteration, sources, expected):
        super(TraversalResp, self).__init__(src, dst, iteration)
        self.sources = sources
        self.expected = expected

    def __str__(self):
        return "(TraversalResp) " + super(TraversalResp, self).__str__()

class StartUpdate(Message):
    def __str__(self):
        return "(StartUpdate) " + super(StartUpdate, self).__str__()

class Update(Message):
    def __str__(self):
        return "(Update) " + super(Update, self).__str__()

class RemovalReq(Message):
    def __str__(self):
        return "(RemovalReq) " + super(RemovalReq, self).__str__()

class RemovalAck(Message):
    def __str__(self):
        return "(RemovalAck) " + super(RemovalAck, self).__str__()

class Convergence(Message):
    def __str__(self):
        return "(Convergence) " + super(Convergence, self).__str__()

################
## Node Types ##
################

class Node(object):
    """Base Node class. Each Node is a single nanoservice. A Node represents a single body in the simulation."""
    count = 0
    def __init__(self, env, network):
        self.env = env
        self.logger = Logger(env)
        self.network = network
        self.queue = simpy.Store(self.env)
        self.ID = Node.count
        Node.count += 1
        self.iteration_cnt = 0 # local iteration counter

    @staticmethod
    def init_params():
        Node.traversal_req_service_time = Node.sample_generator(NBodySimulator.config['traversal_req_service_time_dist'].next())
        Node.traversal_resp_service_time = Node.sample_generator(NBodySimulator.config['traversal_resp_service_time_dist'].next())
        Node.traversal_prob = NBodySimulator.config['traversal_probability'].next()

    @staticmethod
    def sample_generator(filename):
        # read the file and generate samples
        samples = pd.read_csv(filename)
        while True:
            yield random.choice(samples['all'])


class Particle(Node):
    """Particle Node class."""
    count = 0
    def __init__(self, env, network, parents):
        super(Particle, self).__init__(env, network)
        self.particle_ID = Particle.count
        Particle.count += 1
        self.env.process(self.start())
        self.name = 'P-{}'.format(self.ID)
        self.logger.set_filename('{}.log'.format(self.name))

        self.parents = parents
        self.master_ID = self.ID
        self.response_bitmap = {}

        # add to the list of simulation nodes
        NBodySimulator.all_nodes += self.get_all_replicas()
        NBodySimulator.particle_nodes += self.get_all_replicas()

    def __str__(self):
        return self.name

    @staticmethod
    def init_params():
        pass

    def get_all_replicas(self):
        return [self]

    def log(self, s):
        return self.logger.log('Particle Node {}: iteration_cnt: {} {}'.format(self.ID, self.iteration_cnt, s))

    def start(self):
        """Receive and process messages"""
        while not NBodySimulator.complete:
            # wait for a msg that is for the current iteration
            msg = yield self.queue.get()
            self.log('Processing message: {}'.format(str(msg)))
            if type(msg) == Start:
                # send out initial traversal msg to a root node replica
                dst_node = random.choice(NBodySimulator.root_node.get_all_replicas())
                self.log('Sending traversal msg to root node {}'.format(dst_node.ID))
                self.network.queue.put(TraversalReq(self.ID, dst_node.ID, self.iteration_cnt, self.ID, sources=[], expected=[dst_node.master_ID]))
            elif type(msg) == TraversalReq:
                # process request and send back response
                yield self.env.timeout(Node.traversal_req_service_time.next())
                self.log('Sending traversal response to node {}'.format(msg.traversal_src))
                self.network.queue.put(TraversalResp(self.ID, msg.traversal_src, msg.iteration, sources=[self.master_ID]+msg.sources, expected=msg.expected))
            elif type(msg) == TraversalResp:
                # There will be one Traversal Response from each node that is "sufficiently far away"
                # To process a response update the response_bitmap. All responses have been received
                # once all bits in the bitmap are set.
                yield self.env.timeout(Node.traversal_resp_service_time.next())
                # update response bitmap
                for n in msg.expected:
                    if n not in self.response_bitmap:
                        self.response_bitmap[n] = 0
                for n in msg.sources:
                    self.response_bitmap[n] = 1
                if len(self.response_bitmap) == sum(self.response_bitmap.values()):
                    # all required responses have been received for this particle
                    NBodySimulator.check_traversal_done(self.env.now)
            # TODO: implement tree update phase
            elif type(msg) == StartUpdate:
                # check to make sure all responses have been received
                if len(self.response_bitmap) != sum(self.response_bitmap.values()):
                    self.log('ERROR: All required responses have not been received ... Iteration failed')
                    sys.exit(1)
                # TODO: This is just place holder logic to move onto the next iteration immediately because the update phase is not implemented
                self.iteration_cnt += 1
                self.response_bitmap = {}
                NBodySimulator.check_done(self.env.now)
                if not NBodySimulator.complete:
                    # send next traversal request
                    dst_node = random.choice(NBodySimulator.root_node.get_all_replicas())
                    self.log('Sending traversal msg to root node {}'.format(dst_node.ID))
                    self.network.queue.put(TraversalReq(self.ID, dst_node.ID, self.iteration_cnt, self.ID, sources=[], expected=[dst_node.master_ID]))
#            # TODO: currently unused
#            elif type(msg) == Convergence:
#                # quad tree has converged and it is safe to move onto the next iteration
#                self.iteration_cnt += 1
#                # check if the simulation has completed
#                NBodySimulator.check_done(self.env.now)
#                if not NBodySimulator.complete:
#                    # send next traversal request
#                    dst_node = random.choice(NBodySimulator.root_node.get_all_replicas())
#                    self.log('Sending traversal msg to root node: {}'.format(dst_node.ID))
#                    self.network.queue.put(TraversalReq(self.ID, dst_node.ID, self.iteration_cnt))
            else:
               self.log('ERROR: Received unexpected message: {}'.format(str(msg)))
               sys.exit(1)


class InternalReplicaNode(Node):
    """Internal Replica Node class."""
    count = 0
    def __init__(self, env, network, parents, master_ID=None):
        super(InternalReplicaNode, self).__init__(env, network)
        self.internal_node_ID = InternalReplicaNode.count
        InternalReplicaNode.count += 1
        self.env.process(self.start())
        self.name = 'IR-{}'.format(self.ID)
        self.logger.set_filename('{}.log'.format(self.name))

        self.parents = parents

        if master_ID is None:
            self.master_ID = self.ID
        else:
            self.master_ID = master_ID

    @staticmethod
    def init_params():
        pass

    def log(self, s):
        return self.logger.log('Internal Replication Node {}: iteration_cnt: {} {}'.format(self.ID, self.iteration_cnt, s))

    def set_children(self, children):
        self.children = children

    def start(self):
        """Receive and process messages"""
        while not NBodySimulator.complete:
            # wait for a msg that is for the current iteration
            msg = yield self.queue.get()
            self.log('Processing message: {}'.format(str(msg)))
            if type(msg) == TraversalReq:
                # There is some probability that this node will service the request (i.e. the particle is
                # sufficiently far away). In this case the node will send back a response. Otherwise, the
                # node will forward the traversal request to every child (one replica each).
                if random.random() < Node.traversal_prob:
                    # this node is "sufficiently far away" and can respond
                    self.network.queue.put(TraversalResp(self.ID, msg.traversal_src, msg.iteration, sources=[self.master_ID]+msg.sources, expected=msg.expected))
                else:
                    # need to forward traversal request to one replica of each child
                    # select targets
                    targets = []
                    for c, replicas in self.children.iteritems():
                        targets.append(random.choice(replicas))
                    master_targets = [t.master_ID for t in targets]
                    # forward request to each target
                    for t in targets:
                        fwd_msg = TraversalReq(self.ID, t.ID, msg.iteration, msg.traversal_src,
                                               sources = [self.master_ID] + msg.sources,
                                               expected = msg.expected + master_targets)
                        self.network.queue.put(fwd_msg)
#            # TODO: implement the tree update phase
#            elif type(msg) == Update:
#                pass
#            elif type(msg) == RemovalReq:
#                pass
#            elif type(msg) == RemovalResp:
#                pass
#            elif type(msg) == Convergence:
#                pass
            else:
               self.log('ERROR: Received unexpected message: {}'.format(str(msg)))
               sys.exit(1)

class InternalMasterNode(InternalReplicaNode):
    """Internal Master Node class."""
    count = 0
    def __init__(self, env, network, quadtree, parents):
        super(InternalMasterNode, self).__init__(env, network, parents)
        self.internal_node_ID = InternalMasterNode.count
        InternalMasterNode.count += 1
#        self.env.process(self.start())
        self.name = 'IM-{}'.format(self.ID)
        self.logger.set_filename('{}.log'.format(self.name))

        # make replicas
        self.replicas = [InternalReplicaNode(env, network, parents, master_ID=self.ID) for i in range(InternalMasterNode.rep_factor-1)]

        # make sure that this is actually an internal node
        assert len(quadtree.children) > 0

        # create the children nodes
        # self.children is a dictionary that maps position to list of replicas
        self.children = {'NW':[], 'NE':[], 'SW':[], 'SE':[]}
        for pos, qt_child in quadtree.children.iteritems():
            if qt_child.num_particles() == 1:
                # create particle node
                node = Particle(env, network, parents=self.get_all_replicas())
            else:
                # create new internal master node (recursive call)
                node = InternalMasterNode(env, network, qt_child, parents=self.get_all_replicas())
            self.children[pos] = node.get_all_replicas()

        # assign children to all replicas
        for r in self.replicas:
            r.set_children(self.children)

        # add self and replicas to simulation nodes
        NBodySimulator.all_nodes += self.get_all_replicas()
        NBodySimulator.internal_nodes += self.get_all_replicas()

    def str_help(self, pos):
        if len(self.children[pos]) > 0:
            return str([n for n in self.children[pos] if type(n) == InternalMasterNode or type(n) == Particle][0])
        else:
            return ''

    def __str__(self):
        node_str = str([n.name for n in self.get_all_replicas()])
        nw_str = self.str_help('NW')
        ne_str = self.str_help('NE')
        sw_str = self.str_help('SW')
        se_str = self.str_help('SE')
        return """{}
NW: ({})
NE: ({})
SW: ({})
SE: ({})
""".format(node_str, nw_str, ne_str, sw_str, se_str)

    def get_all_replicas(self):
        return [self] + self.replicas

    @staticmethod
    def init_params():
        InternalMasterNode.rep_factor = NBodySimulator.config['replication_factor'].next()

    def log(self, s):
        return self.logger.log('{}: iteration_cnt: {} {}'.format(self.name, self.iteration_cnt, s))

#     def start(self):
#         """Receive and process messages"""
#         while not NBodySimulator.complete:
#             # wait for a msg that is for the current iteration
#             msg = yield self.queue.get()
#             self.log('Processing message: {}'.format(str(msg)))
#             if type(msg) == TraversalReq:
#                 # There is some probability that this node will service the request (i.e. the particle is
#                 # sufficiently far away). In this case the node will send back a response. Otherwise, the
#                 # node will forward the traversal request to every child (one replica each).
#                 if random.random() < Node.traversal_prob:
#                     # this node is "sufficiently far away" and can respond
#                     self.network.queue.put(TraversalResp(self.ID, msg.traversal_src, msg.iteration, sources=[self.master_ID]+msg.sources, expected=msg.expected))
#                 else:
#                     # need to forward traversal request to one replica of each child
#                     # select targets
#                     targets = []
#                     for c, replicas in self.children.iteritems():
#                         targets.append(random.choice(replicas))
#                     master_targets = [t.master_ID for t in targets]
#                     # forward request to each target
#                     for t in targets:
#                         fwd_msg = TraversalReq(self.ID, t.ID, msg.iteration, msg.traversal_src,
#                                                sources = [self.master_ID] + msg.sources,
#                                                expected = msg.expected + master_targets)
#                         self.network.queue.put(fwd_msg)
# #            # TODO: implement tree update logic ...
# #            elif type(msg) == Update:
# #                # The update msg either came from a parent (new particle has entered the bounding box)
# #                # or came from a child (is this particle still in the bounding box?)
# #                if msg.src in self.parents:
# #                    # classify the particle into one of the children
# #                    child_name = random.choice(self.children.keys())
# #                    if child_name.startswith('P'):
# #                        child_node = self.children[child_name][0]
# #                        # the selected child is a particle node so need to spawn a new InternalMasterNode
# #                        del self.children[child_name]
# #                        new_node = InternalMasterNode(self.env, Logger(self.env), self.network, QuadTree(children=[msg.particle_node, child_node]), parents=self.get_all_replicas())
# #                        self.children[new_node.name] = new_node.get_all_replicas()
# #                        # TODO: need to add this new node to the global list of nodes and start sampling its queue size
# #                        # TODO: send out ReplicaUpdate msg to inform them of the new child
# #                else:
# #                    # assume the msg came from a child node
# #            elif type(msg) == RemovalReq:
# #                pass 
# #            elif type(msg) == RemovalResp:
# #                pass
# #            elif type(msg) == Convergence:
# #                pass
#             else:
#                self.log('ERROR: Received unexpected message: {}'.format(str(msg)))
#                sys.exit(1)


#############
## Network ##
#############

class Network(object):
    """Network which moves messages between nodes"""
    def __init__(self, env):
        self.env = env
        self.logger = Logger(env)
        self.logger.set_filename('network.log')
        self.queue = simpy.Store(env)
        self.nodes = {}
        self.env.process(self.start())

    @staticmethod
    def init_params():
        Network.delay = NBodySimulator.config['network_delay'].next()

    def add_nodes(self, nodes):
        self.nodes = {n.ID: n for n in nodes}

    def start(self):
        """Start forwarding messages"""
        while not NBodySimulator.complete:
            msg = yield self.queue.get()
            self.logger.log('Switching msg\n\t"{}"'.format(str(msg)))
            # need to kick this off asynchronously so this is a non-blocking network
            self.env.process(self.transmit_msg(msg))

    def transmit_msg(self, msg):
        # model the network communication delay
        yield self.env.timeout(Network.delay)
        # put the message in the node's queue
        self.nodes[msg.dst].queue.put(msg)


def DistGenerator(dist, **kwargs):
    if dist == 'bimodal':
        bimodal_samples = map(int, list(np.random.normal(kwargs['lower_mean'], kwargs['lower_stddev'], kwargs['lower_samples']))
                                   + list(np.random.normal(kwargs['upper_mean'], kwargs['upper_stddev'], kwargs['upper_samples'])))
    while True:
        if dist == 'uniform':
            yield random.randint(kwargs['min'], kwargs['max'])
        elif dist == 'normal':
            yield int(np.random.normal(kwargs['mean'], kwargs['stddev']))
        elif dist == 'poisson':
            yield np.random.poisson(kwargs['lambda']) 
        elif dist == 'lognormal':
            yield int(np.random.lognormal(kwargs['mean'], kwargs['sigma']))
        elif dist == 'exponential':
            yield int(np.random.exponential(kwargs['lambda']))
        elif dist == 'fixed':
            yield kwargs['value']
        elif dist == 'bimodal':
            yield random.choice(bimodal_samples)
        else:
            print 'ERROR: Unsupported distrbution: {}'.format(dist)
            sys.exit(1)

###############
## Quad Tree ##
###############
class QuadTree:
    """Simple Quad-tree class.
       Adapted from: https://www.astroml.org/book_figures/chapter2/fig_quadtree_example.html
    """

    # class initialization function
    def __init__(self, data, mins, maxs, parent=None):
        self.parent = parent
        self.data = np.asarray(data)

        # data should be two-dimensional
        assert self.data.shape[1] == 2

        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.children = {}

        mids = 0.5 * (self.mins + self.maxs)
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = mids

        # keep dividing if there is more than one particle in the box
        if self.num_particles() > 1:
            # split the data into four quadrants
            data_q1 = data[(data[:, 0] < mids[0])
                           & (data[:, 1] < mids[1])]
            data_q2 = data[(data[:, 0] < mids[0])
                           & (data[:, 1] >= mids[1])]
            data_q3 = data[(data[:, 0] >= mids[0])
                           & (data[:, 1] < mids[1])]
            data_q4 = data[(data[:, 0] >= mids[0])
                           & (data[:, 1] >= mids[1])]

            # recursively build a quad tree on each quadrant which has data
            if data_q1.shape[0] > 0:
                self.children['SW'] = QuadTree(data_q1,
                                               [xmin, ymin], [xmid, ymid],
                                               self)
            if data_q2.shape[0] > 0:
                self.children['NW'] = QuadTree(data_q2,
                                               [xmin, ymid], [xmid, ymax],
                                               self)
            if data_q3.shape[0] > 0:
                self.children['SE'] = QuadTree(data_q3,
                                               [xmid, ymin], [xmax, ymid],
                                               self)
            if data_q4.shape[0] > 0:
                self.children['NE'] = QuadTree(data_q4,
                                               [xmid, ymid], [xmax, ymax],
                                               self)

    def num_particles(self):
        return self.data.shape[0]

    def draw_rectangle(self, ax):
        """Recursively plot every box with exactly one particle"""
        if self.num_particles() == 1:
            rect = plt.Rectangle(self.mins, *self.sizes, zorder=2,
                                 ec='#000000', fc='none')
            ax.add_patch(rect)
        if self.num_particles() > 1:
            for pos, child in self.children.iteritems():
                child.draw_rectangle(ax)

######################
## N-Body Simulator ##
######################

class NBodySimulator(object):
    """This class controls the simulation"""
    config = {} # user specified input
    out_dir = 'out'
    out_run_dir = 'out/run-0'
    # global logs (across runs)
    avg_throughput = {'all':[]} # iterations/time
    def __init__(self, env):
        self.env = env
        NBodySimulator.num_iterations = NBodySimulator.config['num_iterations'].next()
        NBodySimulator.num_particles = NBodySimulator.config['num_particles'].next()
        NBodySimulator.sample_period = NBodySimulator.config['sample_period'].next()
        self.logger = Logger(env)
        self.logger.set_filename('n-body-sim.log')
        self.network = Network(self.env)

        Message.count = 0
        Node.count = 0
        Particle.count = 0
        InternalReplicaNode.count = 0
        InternalMasterNode.count = 0

        # create quad tree
        NBodySimulator.all_nodes = []
        NBodySimulator.particle_nodes = []
        NBodySimulator.internal_nodes = []
        self.make_quadtree()

        # connect nodes to network
        self.network.add_nodes(NBodySimulator.all_nodes)
        
        self.init_sim()

    def make_quadtree(self):
        """
        Create a random quad tree with the appropriate number of particles.
        """
        X = np.random.random((NBodySimulator.num_particles, 2))
        mins = (-0.1, -0.1)
        maxs = (1.1, 1.1)
        quadtree = QuadTree(X, mins, maxs)
        NBodySimulator.root_node = InternalMasterNode(self.env, self.network, quadtree, parents=[])
        print "Quad Tree:"
        print str(NBodySimulator.root_node)

    def init_sim(self):
        # initialize run local variables
        self.q_sizes = {n.name:[] for n in NBodySimulator.all_nodes}
        self.q_sizes['time'] = []
        self.q_sizes['network'] = []
        NBodySimulator.complete = False
        NBodySimulator.traversed_node_cnt = 0
        NBodySimulator.converged_node_cnt = 0
        NBodySimulator.iteration_cnt = 0
        NBodySimulator.traversal_times = {'all': []}
        NBodySimulator.iteration_times = {'all': []}
        NBodySimulator.traversal_start_time = 0
        NBodySimulator.iteration_start_time = 0
        NBodySimulator.finish_time = 0
        # send out Start messages to particle nodes
        for n in NBodySimulator.particle_nodes:
            n.queue.put(Start(0, n.ID, 0))
        # start logging
        if self.sample_period > 0:
            self.env.process(self.sample_queues())

    def sample_queues(self):
        """Sample node queue occupancy"""
        while not NBodySimulator.complete:
            self.q_sizes['time'].append(self.env.now)
            self.q_sizes['network'].append(len(self.network.queue.items))
            for n in NBodySimulator.all_nodes:
                self.q_sizes[n.name].append(len(n.queue.items))
            yield self.env.timeout(NBodySimulator.sample_period)

    @staticmethod
    def check_traversal_done(now):
        """This is called by each particle after it has received all of its required responses in the
           traversal phase of each iteration
        """
        NBodySimulator.traversed_node_cnt += 1
        if NBodySimulator.traversed_node_cnt == len(NBodySimulator.particle_nodes):
            # all particles have completed traversal
            NBodySimulator.traversal_times['all'].append(now - NBodySimulator.traversal_start_time)
            NBodySimulator.traversal_start_time = now
            NBodySimulator.traversed_node_cnt = 0
            # TODO: this is temporary. The StartUpdate msgs will be triggered using a timer
            # send out StartUpdate messages to all particle nodes
            for n in NBodySimulator.particle_nodes:
                n.queue.put(StartUpdate(0, n.ID, NBodySimulator.iteration_cnt))

    @staticmethod
    def check_done(now):
        """This is called by each particle node after receiving convergence message in each iteration"""
        NBodySimulator.converged_node_cnt += 1
        # increment the iteration_cnt if all particle nodes have converged
        if NBodySimulator.converged_node_cnt == len(NBodySimulator.particle_nodes):
            NBodySimulator.iteration_times['all'].append(now - NBodySimulator.iteration_start_time)
            NBodySimulator.iteration_start_time = now
            NBodySimulator.iteration_cnt += 1
            NBodySimulator.converged_node_cnt = 0
        # simulation is complete after all iterations
        if NBodySimulator.iteration_cnt == NBodySimulator.num_iterations:
            NBodySimulator.complete = True
            NBodySimulator.finish_time = now

    def dump_run_logs(self):
        """Dump any logs recorded during this run of the simulation"""
        out_dir = os.path.join(os.getcwd(), NBodySimulator.out_run_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # log the measured queue sizes
        df = pd.DataFrame(self.q_sizes)
        write_csv(df, os.path.join(NBodySimulator.out_run_dir, 'q_sizes.csv'))

        # log the traversal times
        df = pd.DataFrame(NBodySimulator.traversal_times)
        write_csv(df, os.path.join(NBodySimulator.out_run_dir, 'traversal_times.csv'))

        # log the iteration times
        df = pd.DataFrame(NBodySimulator.iteration_times)
        write_csv(df, os.path.join(NBodySimulator.out_run_dir, 'iteration_times.csv'))

        # record avg throughput for this run in terms of iterations/microsecond
        throughput = 1e3*float(NBodySimulator.num_iterations)/(NBodySimulator.finish_time)
        NBodySimulator.avg_throughput['all'].append(throughput)

    @staticmethod
    def dump_global_logs():
        # log avg throughput
        df = pd.DataFrame(NBodySimulator.avg_throughput)
        write_csv(df, os.path.join(NBodySimulator.out_dir, 'avg_throughput.csv'))

def write_csv(df, filename):
    with open(filename, 'w') as f:
            f.write(df.to_csv(index=False))

def param(x):
    while True:
        yield x

def param_list(L):
    for x in L:
        yield x

def parse_config(config_file):
    """ Convert each parameter in the JSON config file into a generator
    """
    with open(config_file) as f:
        config = json.load(f)

    for p, val in config.iteritems():
        if type(val) == list:
            config[p] = param_list(val)
        else:
            config[p] = param(val)

    return config

def run_nbody_sim(cmdline_args):
    NBodySimulator.config = parse_config(cmdline_args.config)
    # make sure output directory exists
    NBodySimulator.out_dir = NBodySimulator.config['out_dir'].next()
    out_dir = os.path.join(os.getcwd(), NBodySimulator.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # copy config file into output directory
    os.system('cp {} {}'.format(cmdline_args.config, out_dir))
    # run the simulations
    run_cnt = 0
    try:
        while True:
            print 'Running simulation {} ...'.format(run_cnt)
            # initialize random seed
            random.seed(1)
            np.random.seed(1)
            # init params for this run on all classes
            Logger.init_params()
            Message.init_params()
            Node.init_params()
            Particle.init_params()
            InternalReplicaNode.init_params()
            InternalMasterNode.init_params()
            Network.init_params()
            NBodySimulator.out_run_dir = os.path.join(NBodySimulator.out_dir, 'run-{}'.format(run_cnt))
            if not os.path.exists(NBodySimulator.out_run_dir):
                os.makedirs(NBodySimulator.out_run_dir)
            run_cnt += 1
            env = simpy.Environment()
            s = NBodySimulator(env)
            env.run()
            s.dump_run_logs()
    except StopIteration:
        NBodySimulator.dump_global_logs()
        print 'All Simulations Complete!'

def main():
    args = cmd_parser.parse_args()
    # Run the simulation
    run_nbody_sim(args)

if __name__ == '__main__':
    main()

