#!/usr/bin/env python2

import argparse
import simpy
import pandas as pd
import numpy as np
import sys, os
import random
import json
from collections import OrderedDict
from asciitree import LeftAligned

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
        if Logger.debug:
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

class StartTraversal(Message):
    def __str__(self):
        return "(StartTraversal) " + super(StartTraversal, self).__str__()

class TraversalReq(Message):
    def __init__(self, src, dst, iteration, traversal_src, pos, sources, expected):
        super(TraversalReq, self).__init__(src, dst, iteration)
        self.pos = pos # position of particle
        self.traversal_src = traversal_src
        self.sources = sources
        self.expected = expected

    def __str__(self):
        return "(TraversalReq) " + super(TraversalReq, self).__str__() + ", traversal_src: {}, pos: {}, sources: {}, expected: {}".format(self.traversal_src, self.pos, self.sources, self.expected)

class TraversalResp(Message):
    def __init__(self, src, dst, iteration, sources, expected):
        super(TraversalResp, self).__init__(src, dst, iteration)
        self.sources = sources
        self.expected = expected

    def __str__(self):
        return "(TraversalResp) " + super(TraversalResp, self).__str__() + ", sources: {}, expected: {}".format(self.sources, self.expected)

class StartUpdate(Message):
    def __str__(self):
        return "(StartUpdate) " + super(StartUpdate, self).__str__()

class Update(Message):
    def __init__(self, src, dst, iteration, update_src, pos):
        super(Update, self).__init__(src, dst, iteration)
        self.update_src = update_src # ID of the particle that initiated the update 
        self.pos = pos # position of particle

    def __str__(self):
        return "(Update) " + super(Update, self).__str__() + ", update_src: {}, pos: {}".format(self.update_src, self.pos)

# NOTE: currently unused
class RemovalReq(Message):
    def __str__(self):
        return "(RemovalReq) " + super(RemovalReq, self).__str__()

# NOTE: currently unused
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

    @staticmethod
    def init_params():
        Node.traversal_req_service_time = Node.sample_generator(NBodySimulator.config['traversal_req_service_time_dist'].next())
        Node.traversal_resp_service_time = Node.sample_generator(NBodySimulator.config['traversal_resp_service_time_dist'].next())
        Node.theta = NBodySimulator.config['theta'].next()
        Node.velocity = NBodySimulator.config['velocity'].next()

    @staticmethod
    def sample_generator(filename):
        # read the file and generate samples
        samples = pd.read_csv(filename)
        while True:
            yield random.choice(samples['all'])


class Particle(Node):
    """Particle Node class."""
    count = 0
    def __init__(self, env, network, pos, parents):
        super(Particle, self).__init__(env, network)
        self.iteration_cnt = 0 # local iteration counter
        self.particle_ID = Particle.count
        Particle.count += 1
        self.env.process(self.start())
        self.name = 'P-{}'.format(self.ID)
        self.logger.set_filename('{}.log'.format(self.name))

        # (X,Y) position of particle
        self.pos = pos

        # Send StartUpdate msg to neighbor after we finish updating the tree
        self.neighbor = None # default

        self.parents = parents
        self.master_ID = self.ID
        self.response_bitmap = {}

        # add to the list of simulation nodes
        for n in self.get_all_replicas():
            NBodySimulator.nodes[n.ID] = n
            NBodySimulator.particle_nodes[n.ID] = n

    def __str__(self):
        return self.name

    def make_dict(self):
        return OrderedDict()

    @staticmethod
    def init_params():
        pass

    def set_neighbor(self, neighbor):
        self.neighbor = neighbor

    def set_parents(self, parents):
        self.parents = parents

    def get_all_replicas(self):
        return [self]

    def log(self, s):
        return self.logger.log('{}: iteration_cnt: {} {}'.format(self.name, self.iteration_cnt, s))

    def start(self):
        """Receive and process messages"""
        while not NBodySimulator.complete:
            # wait for a msg that is for the current iteration
            msg = yield self.queue.get()
            self.log('Processing message: {}'.format(str(msg)))
            if type(msg) == StartTraversal:
                # send out initial traversal msg to a root node replica
                self.start_traversal()
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
                self.log('Updated response_bitmap = {}'.format(self.response_bitmap))
                if len(self.response_bitmap) == sum(self.response_bitmap.values()):
                    # all required responses have been received for this particle
                    NBodySimulator.check_traversal_done(self.env.now)
            elif type(msg) == StartUpdate:
                # check to make sure all traversal responses have been received
                if len(self.response_bitmap) != sum(self.response_bitmap.values()):
                    self.log('ERROR: All required responses have not been received ... Iteration failed')
                    sys.exit(1)
                # reset traversal response tracking state
                self.response_bitmap = {}
                # update the particle's position randomly
                self.pos += (np.random.random((1, 2))[0] + np.array([-0.5, -0.5]))*Node.velocity
                # make sure the particle stays "in bounds"
                self.pos = self.pos if (self.pos[0] >= 0 and self.pos[0] <= 1 and self.pos[1] >= 0 and self.pos[1] < 1) else np.random.random((1,2))[0]
                # send Update msg to master parent
                self.network.queue.put(Update(self.ID, self.parents[0].master_ID, self.iteration_cnt, self.ID, self.pos))
            elif type(msg) == Convergence:
                # quad tree has converged after our update and its safe to move onto the next (if any) update
                self.iteration_cnt += 1
                # check if the simulation has completed
                NBodySimulator.check_done(self.env.now)
                if self.neighbor is not None:
                    # Our update is complete so it's safe for our neighbor to perform their update
                    self.network.queue.put(StartUpdate(self.ID, self.neighbor.ID, self.iteration_cnt-1))
                else:
                    # this is the last particle that needs to perform an update -- send out StartTraversal msgs if the simulation is not complete
                    if not NBodySimulator.complete:
                        for n in NBodySimulator.particle_nodes.values():
                            self.network.queue.put(StartTraversal(self.ID, n.ID, self.iteration_cnt))
            else:
               self.log('ERROR: Received unexpected message: {}'.format(str(msg)))
               sys.exit(1)

    def start_traversal(self):
        # send next traversal request
        dst_node = random.choice(NBodySimulator.root_node.get_all_replicas())
        self.log('Sending traversal msg to root node: {}'.format(dst_node.ID))
        self.network.queue.put(TraversalReq(self.ID, dst_node.ID, self.iteration_cnt, self.ID, self.pos, sources=[], expected=[dst_node.master_ID]))

class InternalReplicaNode(Node):
    """Internal Replica Node class."""
    def __init__(self, env, network, mins, maxs, parents, master_ID=None):
        super(InternalReplicaNode, self).__init__(env, network)
        self.env.process(self.start())
        self.name = 'IR-{}'.format(self.ID)
        self.logger.set_filename('{}.log'.format(self.name))
        self.killed = False

        self.children = {'NW':[], 'NE':[], 'SW':[], 'SE':[]}
        self.parents = parents
        self.mins = mins
        self.maxs = maxs
        # The midpoint of the bounding box (X,Y) position
        self.mids = 0.5 * (self.mins + self.maxs)
        # Center of Mass (X,Y) position
        # NOTE: the com is actually mids. Is it worth properly computing the center of mass?
        self.com = self.mids
        # Size of bounding box (X width, Y width)
        self.sizes = self.maxs - self.mins

        if master_ID is None:
            self.master_ID = self.ID
        else:
            self.master_ID = master_ID

    @staticmethod
    def init_params():
        pass

    def log(self, s):
        return self.logger.log('{}: {}'.format(self.name, s))

    def process_traversal_req(self, msg):
        yield self.env.timeout(Node.traversal_req_service_time.next())
        # If the particle is sufficiently far away then send back a response. Otherwise, the
        # node will forward the traversal request to every child (one replica each).
        #if random.random() < Node.theta:
        if np.average(self.sizes)/np.linalg.norm(msg.pos - self.com) < Node.theta:
            # this node is "sufficiently far away" and can respond
            self.log('Sending traversal response to particle P-{}'.format(msg.traversal_src))
            self.network.queue.put(TraversalResp(self.ID, msg.traversal_src, msg.iteration, sources=[self.master_ID]+msg.sources, expected=msg.expected))
        else:
            # need to forward traversal request to one replica of each child
            # select targets
            targets = []
            for c, replicas in self.children.iteritems():
                if len(replicas) > 0:
                    targets.append(random.choice(replicas))
            master_targets = [t.master_ID for t in targets]
            self.log('Sending traversal requests to nodes: {}'.format(str([n.name for n in targets])))
            # forward request to each target
            for t in targets:
                fwd_msg = TraversalReq(self.ID, t.ID, msg.iteration, msg.traversal_src, msg.pos,
                                       sources = [self.master_ID] + msg.sources,
                                       expected = msg.expected + master_targets)
                self.network.queue.put(fwd_msg)

    def kill(self):
        self.killed = True
        del NBodySimulator.nodes[self.ID]

    def start(self):
        """Receive and process messages"""
        while not NBodySimulator.complete and not self.killed:
            # wait for a msg that is for the current iteration
            msg = yield self.queue.get()
            self.log('Processing message: {}'.format(str(msg)))
            if type(msg) == TraversalReq:
                yield self.env.process(self.process_traversal_req(msg))
            else:
               self.log('ERROR: Received unexpected message: {}'.format(str(msg)))
               sys.exit(1)

class InternalMasterNode(InternalReplicaNode):
    """Internal Master Node class."""
    def __init__(self, env, network, mins, maxs, particles=[], parents=[]):
        super(InternalMasterNode, self).__init__(env, network, mins, maxs, parents)
        self.name = 'IM-{}'.format(self.ID)
        self.logger.set_filename('{}.log'.format(self.name))

        # make replicas
        self.replicas = [InternalReplicaNode(env, network, mins, maxs, parents, master_ID=self.ID) for i in range(InternalMasterNode.rep_factor-1)]

        # add self and replicas to simulation nodes
        for n in self.get_all_replicas():
            NBodySimulator.nodes[n.ID] = n

        if len(particles) > 0:
            data = np.array([p.pos for p in particles])

            # split the particle positions into four quadrants
            data_q1 = data[(data[:, 0] < self.mids[0])
                           & (data[:, 1] < self.mids[1])]
            data_q2 = data[(data[:, 0] < self.mids[0])
                           & (data[:, 1] >= self.mids[1])]
            data_q3 = data[(data[:, 0] >= self.mids[0])
                           & (data[:, 1] < self.mids[1])]
            data_q4 = data[(data[:, 0] >= self.mids[0])
                           & (data[:, 1] >= self.mids[1])]

            # recursively build a quad tree on each quadrant with multiple particles
            q1_particles = [p for p in particles if p.pos in data_q1]
            self.build_subtree('SW', data_q1, self.get_child_mins('SW'), self.get_child_maxs('SW'), q1_particles)

            q2_particles = [p for p in particles if p.pos in data_q2]
            self.build_subtree('NW', data_q2, self.get_child_mins('NW'), self.get_child_maxs('NW'), q2_particles)

            q3_particles = [p for p in particles if p.pos in data_q3]
            self.build_subtree('SE', data_q3, self.get_child_mins('SE'), self.get_child_maxs('SE'), q3_particles)

            q4_particles = [p for p in particles if p.pos in data_q4]
            self.build_subtree('NE', data_q4, self.get_child_mins('NE'), self.get_child_maxs('NE'), q4_particles)

            # assign children to all replicas
            for r in self.replicas:
                r.children = self.children

    def get_child_mins(self, box):
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = self.mids

        if box == 'SW':
            return np.array([xmin, ymin])
        elif box == 'NW':
            return np.array([xmin, ymid])
        elif box == 'SE':
            return np.array([xmid, ymin])
        elif box == 'NE':
            return np.array([xmid, ymid])

    def get_child_maxs(self, box):
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = self.mids

        if box == 'SW':
            return np.array([xmid, ymid])
        elif box == 'NW':
            return np.array([xmid, ymax])
        elif box == 'SE':
            return np.array([xmax, ymid])
        elif box == 'NE':
            return np.array([xmax, ymax])

    def build_subtree(self, box, data, mins, maxs, particles):
        # recursively build a quad tree on each quadrant with multiple particles
        if data.shape[0] > 1:
            # there are multiple particles in this box
            new_node = InternalMasterNode(self.env, self.network, mins, maxs, particles,
                                          parents=self.get_all_replicas())
            self.children[box] = new_node.get_all_replicas()
        elif data.shape[0] == 1:
            # there is only one particle in this box
            assert len(particles) == 1, "Incorrect number of particles in box ..."
            self.children[box] = particles[0].get_all_replicas()
            particles[0].set_parents(self.get_all_replicas())
        else:
            self.children[box] = []

    def get_all_replicas(self):
        return [self] + self.replicas

    def make_dict(self):
        d = OrderedDict()
        for box, children in self.children.iteritems():
            if len(children) > 0:
                c = [c for c in children if type(c) == InternalMasterNode or type(c) == Particle][0]
                d['{}::{}'.format(box, str(c))] = c.make_dict()
        return d

    def __str__(self):
        return str([n.name for n in self.get_all_replicas()])

    @staticmethod
    def init_params():
        InternalMasterNode.rep_factor = NBodySimulator.config['replication_factor'].next()

    def lookup_child_ID(self, ID):
        """Return the box ID (i.e. NW, NE, SE, SW) for the given node ID, or None"""
        for box_ID, children in self.children.iteritems():
            if ID in [c.ID for c in children]:
                return box_ID
        return None

    def kill(self):
        # cannot kill the root node
        assert len(self.parents) > 0, "Cannot kill the root node" 
        self.log('Killed')

        self.killed = True
        del NBodySimulator.nodes[self.ID]

        for node in self.replicas:
            node.kill()

        parent_master = NBodySimulator.nodes[self.parents[0].master_ID]
        parent_box_ID = parent_master.lookup_child_ID(self.ID)

        # move any particle children up as the parents new children
        for box, c_replicas in self.children.iteritems():
            if len(c_replicas) > 0 and type(c_replicas[0]) == Particle:
                # assign parents new children
                for p in self.parents:
                    p.children[parent_box_ID] = c_replicas
                # assign children's new parents
                for c in c_replicas:
                    c.set_parents(self.parents)
        # check if the parent should also be killed
        parent_ID = parent_master.check_kill()

        # return the ID of the parent of the last killed node
        parent_ID = parent_ID if parent_ID is not None else self.parents[0].master_ID
        return parent_ID

    def start(self):
        """Receive and process messages"""
        while not NBodySimulator.complete and not self.killed:
            # wait for a msg that is for the current iteration
            msg = yield self.queue.get()
            self.log('Processing message: {}'.format(str(msg)))
            if type(msg) == TraversalReq:
                yield self.env.process(self.process_traversal_req(msg))
            elif type(msg) == Update:
                # check if the particle is in this bounding box
                result = self.classify_pos(msg.pos)
                self.log('Classified particle P-{} ({}) into box {}'.format(msg.update_src, msg.pos, result))
                if result is None:
                    # the particle is not in this bounding box
                    assert len(self.parents) > 0, "Particle moved out of bounds!"
                    # this is not the root node
                    parent_ID = None
                    child_box = self.lookup_child_ID(msg.src)
                    if child_box is not None and type(NBodySimulator.nodes[msg.src]) == Particle:
                        # msg came from a child particle -- remove it as a child
                        self.set_child(child_box, [])
                        parent_ID = self.check_kill()
                    # send update to parent
                    parent_ID = parent_ID if parent_ID is not None else self.parents[0].master_ID
                    self.network.queue.put(Update(self.ID, parent_ID, msg.iteration, msg.update_src, msg.pos))
                else:
                    # the particle is in this bounding box
                    particle_node = NBodySimulator.nodes[msg.update_src]
                    self.add_particle(particle_node, msg.iteration, msg.update_src)
            else:
                self.log('ERROR: Received unexpected message: {}'.format(str(msg)))
                sys.exit(1)

    def set_child(self, box, children):
        for n in self.get_all_replicas():
            n.children[box] = children

    def num_children(self):
        return sum([1 for nodes in self.children.values() if len(nodes) > 0])

    def check_kill(self):
        self.log('Checking if this node should be killed')
        parent_ID = None
        num_particle_children = sum([1 for nodes in self.children.values() if (len(nodes) > 0 and type(nodes[0]) == Particle)])
        if num_particle_children == 1 and self.num_children() == 1:
            # there is only one child remaining and it is a particle node -- this node doesn't need to be here anymore
            # remove this node (and replicas) from the tree and the simulation
            parent_ID = self.kill()
        # return the parent of the last killed node (or None if the node is not killed)
        return parent_ID

    def add_particle(self, particle_node, iteration, update_src):
        """Add the given particle node as a child"""
        child_box = self.classify_pos(particle_node.pos)
        assert child_box is not None, "Attempted to add a particle child that is not in this box"

        self.log('Attempting to add particle {} ({}) to box {}'.format(particle_node.name, particle_node.pos, child_box))

        # if the particle is already a child, remove it before preceeding
        orig_child_box = self.lookup_child_ID(particle_node.ID)
        if orig_child_box is not None:
            self.set_child(orig_child_box, [])

        if len(self.children[child_box]) == 0:
            # currently no children is this box so add it here and send convergence msg
            self.log('Added particle {} to box {}'.format(particle_node.pos, child_box))
            self.set_child(child_box, particle_node.get_all_replicas())
            particle_node.set_parents(self.get_all_replicas())
            # send convergence msg to the particle so the next particle can perform its update
            if particle_node.ID == update_src:
                # only send Convergence msg to the particle that initiated the update
                self.network.queue.put(Convergence(self.ID, particle_node.ID, iteration))
        elif type(self.children[child_box][0]) in [InternalMasterNode, InternalReplicaNode]:
            # send update to the child internal node
            self.log('Sending update to child internal node')
            child_master_ID = self.children[child_box][0].master_ID
            self.network.queue.put(Update(self.ID, child_master_ID, iteration, particle_node.ID, particle_node.pos))
        elif type(self.children[child_box][0]) == Particle:
            # particle was classified into the same box as another particle
            self.log('Creating new internal node')
            # create new internal node
            new_node = InternalMasterNode(self.env, self.network, self.get_child_mins(child_box), self.get_child_maxs(child_box), parents=self.get_all_replicas())
            orig_particle_node = NBodySimulator.nodes[self.children[child_box][0].master_ID]
            # recursively add both particles to the new node
            new_node.add_particle(orig_particle_node, iteration, update_src)
            new_node.add_particle(particle_node, iteration, update_src)
            self.set_child(child_box, new_node.get_all_replicas())

    def classify_pos(self, pos):
        """Classify the position into the appropriate chlid bounding box or return none if it is not in this bounding box """

        # first check if the position is in the bounding box
        data = pos[(pos[0] >= self.mids[0] - self.sizes[0]/2.0)
                    & (pos[0] <= self.mids[0] + self.sizes[0]/2.0)
                    & (pos[1] >= self.mids[1] - self.sizes[1]/2.0)
                    & (pos[1] <= self.mids[1] + self.sizes[1]/2.0)]

        if data.shape[0] == 0:
            return None

        # split the data into four quadrants
        data_q1 = pos[(pos[0] < self.mids[0])
                       & (pos[1] < self.mids[1])]
        data_q2 = pos[(pos[0] < self.mids[0])
                       & (pos[1] >= self.mids[1])]
        data_q3 = pos[(pos[0] >= self.mids[0])
                       & (pos[1] < self.mids[1])]
        data_q4 = pos[(pos[0] >= self.mids[0])
                       & (pos[1] >= self.mids[1])]

        if data_q1.shape[0] > 0:
            return 'SW'
        if data_q2.shape[0] > 0:
            return 'NW'
        if data_q3.shape[0] > 0:
            return 'SE'
        if data_q4.shape[0] > 0:
            return 'NE'
        self.log('ERROR: unable to classify position: {}'.format(pos))
        sys.exit(1)

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
        self.env.process(self.start())

    @staticmethod
    def init_params():
        Network.delay = NBodySimulator.config['network_delay'].next()

    def start(self):
        """Start forwarding messages"""
        while not NBodySimulator.complete:
            msg = yield self.queue.get()
            self.logger.log('Switching msg: "{}"'.format(str(msg)))
            # need to kick this off asynchronously so this is a non-blocking network
            self.env.process(self.transmit_msg(msg))

    def transmit_msg(self, msg):
        # model the network communication delay
        yield self.env.timeout(Network.delay)
        # put the message in the node's queue
        # TODO: I think this could potentially fail if a particle node sends an update to a parent that is killed
        NBodySimulator.nodes[msg.dst].queue.put(msg)


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

######################
## N-Body Simulator ##
######################

class NBodySimulator(object):
    """This class controls the simulation"""
    config = {} # user specified input
    out_dir = 'out'
    out_run_dir = 'out/run-0'
    # global logs (across runs)
    run_ID = 0
    run_log = {'run':[],
               'avg_throughput': []} # iterations/time
    def __init__(self, env):
        self.env = env
        NBodySimulator.num_iterations = NBodySimulator.config['num_iterations'].next()
        NBodySimulator.num_particles = NBodySimulator.config['num_particles'].next()
        NBodySimulator.sample_period = NBodySimulator.config['sample_period'].next()
        NBodySimulator.logger = Logger(env)
        NBodySimulator.logger.set_filename('n-body-sim.log')
        self.network = Network(self.env)

        Message.count = 0
        Node.count = 0
        Particle.count = 0

        # initialize run local variables
        self.q_sizes = {}
        self.q_sizes['time'] = []
        self.q_sizes['node'] = []
        self.q_sizes['qsize'] = []
        NBodySimulator.iteration_log = {}
        NBodySimulator.iteration_log['iteration'] = []
        NBodySimulator.iteration_log['traversal_time'] = []
        NBodySimulator.iteration_log['iteration_time'] = []
        NBodySimulator.complete = False
        NBodySimulator.traversed_node_cnt = 0
        NBodySimulator.converged_node_cnt = 0
        NBodySimulator.iteration_cnt = 0
        NBodySimulator.iteration_start_time = 0
        NBodySimulator.finish_time = 0

        # dictionaries to keep track of the currently active nodes & particles
        NBodySimulator.nodes = {}
        NBodySimulator.particle_nodes = {}
        # create quad tree
        self.make_quadtree()

        # send out Start messages to particle nodes
        for n in NBodySimulator.particle_nodes.values():
            self.network.queue.put(StartTraversal(0, n.ID, 0))
        # start logging
        if self.sample_period > 0:
            self.env.process(self.sample_queues())

    @staticmethod
    def log(s):
        NBodySimulator.logger.log('NBodySimulator: run {}: iteration {}: {}'.format(NBodySimulator.run_ID, NBodySimulator.iteration_cnt, s))

    def make_quadtree(self):
        """
        Create a random quad tree with the appropriate number of particles.
        """
        X = np.random.random((NBodySimulator.num_particles, 2))
        mins = np.array([-0.1, -0.1])
        maxs = np.array([1.1, 1.1])
        # create all particles
        particles = [Particle(self.env, self.network, pos, []) for pos in X]
        # assign neighbors to particles
        for i in range(NBodySimulator.num_particles-1):
            particles[i].set_neighbor(particles[i+1])

        NBodySimulator.start_particle = particles[0]

        # create quadtree
        NBodySimulator.root_node = InternalMasterNode(self.env, self.network, mins, maxs, particles, parents=[])
        NBodySimulator.log(NBodySimulator.quadtree_str())

    def sample_queues(self):
        """Sample node queue occupancy"""
        while not NBodySimulator.complete:
            for n in NBodySimulator.nodes.values():
                self.q_sizes['time'].append(self.env.now)
                self.q_sizes['node'].append(n.name)
                self.q_sizes['qsize'].append(len(n.queue.items))
            yield self.env.timeout(NBodySimulator.sample_period)

    @staticmethod
    def quadtree_str():
        d = OrderedDict()
        d['root::{}'.format(str(NBodySimulator.root_node))] = NBodySimulator.root_node.make_dict()
        tr = LeftAligned()
        return 'Quad Tree ({} total nodes):\n'.format(len(NBodySimulator.nodes)) + tr(d)

    @staticmethod
    def check_traversal_done(now):
        """This is called by each particle after it has received all of its required responses in the
           traversal phase of each iteration
        """
        NBodySimulator.traversed_node_cnt += 1
        if NBodySimulator.traversed_node_cnt == len(NBodySimulator.particle_nodes):
            # all particles have completed traversal
            NBodySimulator.iteration_log['iteration'].append(NBodySimulator.iteration_cnt)
            NBodySimulator.iteration_log['traversal_time'].append(now - NBodySimulator.iteration_start_time)
            NBodySimulator.traversed_node_cnt = 0
            # send StartUpdate msg to the start particle
            NBodySimulator.start_particle.queue.put(StartUpdate(0, NBodySimulator.start_particle.ID, NBodySimulator.iteration_cnt))

    @staticmethod
    def check_done(now):
        """This is called by each particle node after receiving convergence message in each iteration"""
        NBodySimulator.converged_node_cnt += 1
        # increment the iteration_cnt if all particle nodes have converged
        if NBodySimulator.converged_node_cnt == len(NBodySimulator.particle_nodes):
            NBodySimulator.iteration_log['iteration_time'].append(now - NBodySimulator.iteration_start_time)
            NBodySimulator.iteration_start_time = now
            NBodySimulator.iteration_cnt += 1
            NBodySimulator.converged_node_cnt = 0
            NBodySimulator.log(NBodySimulator.quadtree_str())
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

        # log the iteration log
        self.log('iteration = {}'.format(NBodySimulator.iteration_log['iteration']))
        self.log('traversal_time = {}'.format(NBodySimulator.iteration_log['traversal_time']))
        self.log('iteration_time = {}'.format(NBodySimulator.iteration_log['iteration_time']))
        df = pd.DataFrame(NBodySimulator.iteration_log)
        write_csv(df, os.path.join(NBodySimulator.out_run_dir, 'iteration_log.csv'))

        # record avg throughput for this run in terms of iterations/microsecond
        throughput = 1e3*float(NBodySimulator.num_iterations)/(NBodySimulator.finish_time)
        NBodySimulator.run_log['run'].append(NBodySimulator.run_ID)
        NBodySimulator.run_log['avg_throughput'].append(throughput)

    @staticmethod
    def dump_global_logs():
        # log avg throughput
        df = pd.DataFrame(NBodySimulator.run_log)
        write_csv(df, os.path.join(NBodySimulator.out_dir, 'run_log.csv'))

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
    NBodySimulator.run_ID = 0
    try:
        while True:
            print 'Running simulation {} ...'.format(NBodySimulator.run_ID)
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
            NBodySimulator.out_run_dir = os.path.join(NBodySimulator.out_dir, 'run-{}'.format(NBodySimulator.run_ID))
            if not os.path.exists(NBodySimulator.out_run_dir):
                os.makedirs(NBodySimulator.out_run_dir)
            env = simpy.Environment()
            s = NBodySimulator(env)
            env.run()
            s.dump_run_logs()
            NBodySimulator.run_ID += 1
    except StopIteration:
        NBodySimulator.dump_global_logs()
        print 'All Simulations Complete!'

def main():
    args = cmd_parser.parse_args()
    # Run the simulation
    run_nbody_sim(args)

if __name__ == '__main__':
    main()

