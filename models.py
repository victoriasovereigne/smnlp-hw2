# models.py

from utils import *
from adagrad_trainer import *
from treedata import *
import numpy as np
import random
import copy

ROOT = -1

# Greedy parsing model. This model treats shift/reduce decisions as a multiclass classification problem.
class GreedyModel(object):
    def __init__(self, feature_indexer, feature_weights, extension):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.extension = extension
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        fi = self.feature_indexer
        weights = self.feature_weights
        state = initial_parser_state(len(sentence))
        decisions = []
        states = []

        label_indexer = get_label_indexer()

        while not state.is_finished():
            log_probs = [np.NINF, np.NINF, np.NINF]
            possible_actions = get_possible_actions(state)

            for i, y_decision in enumerate(possible_actions):
                next_state = state.take_action(y_decision)
                d_id = label_indexer.index_of(y_decision)

                if next_state.is_legal():
                    feats = extract_features(fi, sentence, state, y_decision, False, extension=self.extension)
                    numerator = score_indexed_features(feats, weights)
                    log_probs[d_id] = numerator

            log_probs = log_probs - logaddexp(log_probs)            
            decision = label_indexer.get_object(get_argmax(log_probs))
            next_state = state.take_action(decision)

            if next_state.is_legal():
                states.append(state)
                decisions.append(decision)
                state = next_state

        states.append(state) # add last state
        dependencies = states[-1].get_dep_objs(len(sentence))
        return ParsedSentence(sentence.tokens, dependencies)



# Beam-search-based global parsing model. Shift/reduce decisions are still modeled with local features, but scores are
# accumulated over the whole sequence of decisions to give a "global" decision.
class BeamedModel(object):
    def __init__(self, feature_indexer, feature_weights, beam_size=1, extension=False):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.beam_size = beam_size
        self.extension = extension
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        fi = self.feature_indexer
        weights = self.feature_weights
        state = initial_parser_state(len(sentence))
        beam_size = self.beam_size
        label_indexer = get_label_indexer()
        
        beams = [Beam(beam_size)] * (2*len(sentence) + 1)
        beams[0].add(state, 0)

        for i in xrange(1, 2*len(sentence)+1):
            beams[i] = Beam(beam_size)
            zipped = beams[i-1].get_elts_and_scores()

            for zipp in zipped:
                old_state, old_score = zipp
                possible_actions = get_possible_actions(old_state)

                for y_decision in possible_actions:
                    potential_new_state = old_state.take_action(y_decision)
                    d_id = label_indexer.index_of(y_decision)

                    if potential_new_state.is_legal():
                        new_feats = extract_features(fi, sentence, old_state, y_decision, False, extension=self.extension)
                        new_score = score_indexed_features(new_feats, weights)
                        new_score += old_score
                        beams[i].add(potential_new_state, new_score)

        dependencies = beams[-1].head().get_dep_objs(len(sentence))

        return ParsedSentence(sentence.tokens, dependencies)


# Stores state of a shift-reduce parser, namely the stack, buffer, and the set of dependencies that have
# already been assigned. Supports various accessors as well as the ability to create new ParserStates
# from left_arc, right_arc, and shift.
class ParserState(object):
    # stack and buffer are lists of indices
    # The stack is a list with the top of the stack being the end
    # The buffer is a list with the first item being the front of the buffer (next word)
    # deps is a dictionary mapping *child* indices to *parent* indices
    # (this is the one-to-many map; parent-to-child doesn't work in map-like data structures
    # without having the values be lists)
    def __init__(self, stack, buffer, deps):
        self.stack = stack
        self.buffer = buffer
        self.deps = deps

    def __repr__(self):
        return repr(self.stack) + " " + repr(self.buffer) + " " + repr(self.deps)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.stack == other.stack and self.buffer == other.buffer and self.deps == other.deps
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def stack_len(self):
        return len(self.stack)

    def buffer_len(self):
        return len(self.buffer)

    def is_legal(self):
        return self.stack[0] == -1

    def is_finished(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

    def buffer_head(self):
        return self.get_buffer_word_idx(0)

    # Returns the buffer word at the given index
    def get_buffer_word_idx(self, index):
        if index >= len(self.buffer):
            raise Exception("Can't take the " + repr(index) + " word from the buffer of length " + repr(len(self.buffer)) + ": " + repr(self))
            
        return self.buffer[index]

    # Returns True if idx has all of its children attached already, False otherwise
    def is_complete(self, idx, parsed_sentence):
        _is_complete = True
        for child in xrange(0, len(parsed_sentence)):
            if parsed_sentence.get_parent_idx(child) == idx and (child not in self.deps.keys() or self.deps[child] != idx):
                _is_complete = False
        return _is_complete

    def stack_head(self):
        if len(self.stack) < 1:
            raise Exception("Can't go one back in the stack if there are no elements: " + repr(self))
        return self.stack[-1]

    def stack_two_back(self):
        if len(self.stack) < 2:
            raise Exception("Can't go two back in the stack if there aren't two elements: " + repr(self))
        return self.stack[-2]

    # Returns a new ParserState that is the result of taking the given action.
    # action is a string, either "L", "R", or "S"
    def take_action(self, action):
        if action == "L":
            return self.left_arc()
        elif action == "R":
            return self.right_arc()
        elif action == "S":
            return self.shift()
        else:
            raise Exception("No implementation for action " + action)

    # Returns a new ParserState that is the result of applying left arc to the current state. May crash if the
    # preconditions for left arc aren't met.
    def left_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_two_back(): self.stack_head()})
        new_stack = list(self.stack[0:-2])
        new_stack.append(self.stack_head())
        return ParserState(new_stack, self.buffer, new_deps)

    # Returns a new ParserState that is the result of applying right arc to the current state. May crash if the
    # preconditions for right arc aren't met.
    def right_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_head(): self.stack_two_back()})
        new_stack = list(self.stack[0:-1])
        return ParserState(new_stack, self.buffer, new_deps)

    # Returns a new ParserState that is the result of applying shift to the current state. May crash if the
    # preconditions for right arc aren't met.
    def shift(self):
        new_stack = list(self.stack)
        new_stack.append(self.buffer_head())
        return ParserState(new_stack, self.buffer[1:], self.deps)

    # Return the Dependency objects corresponding to the dependencies added so far to this ParserState
    def get_dep_objs(self, sent_len):
        dep_objs = []
        for i in xrange(0, sent_len):
            dep_objs.append(Dependency(self.deps[i], "?"))
        return dep_objs


# Returns an initial ParserState for a sentence of the given length. Note that because the stack and buffer
# are maintained as indices, knowing the words isn't necessary.
def initial_parser_state(sent_len):
    return ParserState([-1], range(0, sent_len), {})


# Returns an indexer for the three actions so you can iterate over them easily.
def get_label_indexer():
    label_indexer = Indexer()
    label_indexer.get_index("S")
    label_indexer.get_index("L")
    label_indexer.get_index("R")
    return label_indexer

def get_features_state(feature_cache, state_indexer, s_id, state, d_id):
    state_id = state_indexer.index_of(str(state))
    if state_id != -1:
        return feature_cache[s_id][state_id][d_id]
    else:
        return []


def get_possible_actions(state):
    possible_actions = []

    if len(state.stack) < 2 and state.buffer_len() > 0:
        possible_actions = ['S']
    # can only go left or right if buffer is empty
    elif state.buffer_len() == 0:
        # if stack only contains root and another element, right arc
        if len(state.stack) == 2 and state.stack_two_back() == ROOT:
            possible_actions = ['R']
        else:
            possible_actions = ['L', 'R']
    else:
        # root already has child (root can only have 1 child)
        one_back = state.stack_head()
        two_back = state.stack_two_back()

        if two_back == ROOT and ROOT in state.deps.values(): # cannot attach another child to ROOT
            possible_actions = ['S']
        else:
            possible_actions = ['S', 'L', 'R']

    return possible_actions

# if two or more elements are equal, just return a random choice
def get_argmax(argmax):
    if argmax[0] == argmax[1] == argmax[2]:
        return random.choice([0,1,2])
    elif argmax[0] == argmax[1] and argmax[0] > argmax[2]:
        return random.choice([0,1])
    elif argmax[0] == argmax[2] and argmax[0] > argmax[1]:
        return random.choice([0,1])
    elif argmax[1] == argmax[2] and argmax[1] > argmax[0]:
        return random.choice([1,2])
    else:
        return np.argmax(argmax)

def logaddexp(array):
    tmp = array[0]
    for element in array[1:]:
        tmp = np.logaddexp(tmp, element)

    return tmp

# ======================================================
# Returns a GreedyModel trained over the given treebank.
# ======================================================
def train_greedy_model(parsed_sentences, extension=False, mode='SGD'):
    feature_indexer = Indexer()
    label_indexer = get_label_indexer()
    gold_decisions = []
    gold_states = []

    # ======================================================
    # get the gold decisions and gold states
    # ======================================================
    for sentence in parsed_sentences:
        decisions, states = get_decision_sequence(sentence)
        gold_decisions.append(decisions)
        gold_states.append(states)

    # ======================================================
    # extract features
    # ======================================================
    feature_cache = [[[[] for k in xrange(3)] for j in xrange(len(gold_states[i]))] for i in xrange(len(parsed_sentences))]
    state_indexers = []

    for s_id, sentence in enumerate(parsed_sentences):
        if s_id % 100 == 0:
            print "Extracting features:", s_id, "/", len(parsed_sentences)

        state = initial_parser_state(len(sentence))
        curr_gold_decisions = gold_decisions[s_id]
        curr_gold_states = gold_states[s_id]
        state_indexer = Indexer()
        i = 0

        while not state.is_finished():
            gold_decision = curr_gold_decisions[i]
            d_id = label_indexer.index_of(gold_decision)
            possible_actions = get_possible_actions(state)

            for decision in possible_actions:
                d_id = label_indexer.index_of(decision)
                key = (s_id, str(state), d_id)
                tmp_state = state.take_action(decision)

                if tmp_state.is_legal():
                    state_id = state_indexer.get_index(str(state))
                    feats = extract_features(feature_indexer, sentence, state, decision, True, extension=extension)
                    feature_cache[s_id][state_id][d_id] = feats
            
            next_state = state.take_action(gold_decision)
            state = next_state
            i += 1

        state_indexers.append(state_indexer)

    weights = np.zeros(shape=(len(feature_indexer),)) #np.random.rand(len(feature_indexer),)
    ada = AdagradTrainer(weights, lamb=1e-7)

    # ======================================================
    # build a classifier (logistic regression)
    # ======================================================
    num_epochs = 5

    for epoch in xrange(num_epochs):
        for s_id, sentence in enumerate(parsed_sentences):
            if s_id % 100 == 0:
                print "Training:", s_id, "/", len(parsed_sentences)

            state = initial_parser_state(len(sentence))
            decisions = []
            states = []
            sentence_gold_states = gold_states[s_id]
            sentence_gold_decisions = gold_decisions[s_id]
            index = 0 
            state_indexer = state_indexers[s_id]            

            while not state.is_finished():
                possible_actions = get_possible_actions(state)
                log_probs = [np.NINF, np.NINF, np.NINF] 
                # ======================================================
                # calculate numerator and denominator
                # P(y|x) = exp(wT . f(x,y)) / sum(exp(wT . f(x,y'))) over y' in decisions
                # log P(y|x) = (wt . f(x,y)) - logaddexp(wT . f(x,y')) over y'
                # ======================================================
                for i, y_decision in enumerate(possible_actions):
                    d_id = label_indexer.index_of(y_decision)
                    next_state = state.take_action(y_decision)

                    if next_state.is_legal():
                        feats = get_features_state(feature_cache, state_indexer, s_id, state, d_id)
                        numerator = ada.score(feats) #score_indexed_features(feats, weights)
                        log_probs[d_id] = numerator 

                # ======================================================
                # calculate argmax of log probability to make decision
                # ======================================================
                denominator = logaddexp(log_probs)
                log_probs = log_probs - denominator 
                decision = label_indexer.get_object(get_argmax(log_probs))
                gold_decision = sentence_gold_decisions[index]
                next_state = state.take_action(gold_decision)

                # ======================================================
                # compute gradient
                # ======================================================
                gradient = Counter()
                expected = Counter()
                gold_id = label_indexer.index_of(gold_decision)
                fxy_star = get_features_state(feature_cache, state_indexer, s_id, state, gold_id)#get_features_state(feature_cache, s_id, str(state), gold_id)
                gradient.increment_all(fxy_star, 1)

                for d_id2 in xrange(len(label_indexer)):
                    fxy = get_features_state(feature_cache, state_indexer, s_id, state, d_id2)
                    expected.increment_all(fxy, -np.exp(log_probs[d_id2]))

                gradient.add(expected)

                # ======================================================
                # update weight
                # ======================================================
                if mode == 'SGD':
                    for i in gradient.keys():
                        weights[i] += 0.1/(epoch+1) * gradient.get_count(i) # --> best result

                # try adagrad
                if mode == 'ADA':
                    ada.apply_gradient_update(gradient, 1)

                state = next_state
                index += 1

    if mode == 'ADA':
        weights = ada.get_final_weights()
    
    return GreedyModel(feature_indexer, weights, extension)


def compute_successors(old_beam, feature_indexer, sentence, weights, extension):
# def compute_successors(old_beam, feature_cache, state_indexer, s_id, weights):
    new_beam = Beam(old_beam.size)
    zipped = old_beam.get_elts_and_scores()
    label_indexer = get_label_indexer()

    for (old_counter, old_state), old_score in zipped:
        possible_actions = get_possible_actions(old_state)

        for y_decision in possible_actions:
            potential_new_state = old_state.take_action(y_decision)
            # d_id = label_indexer.index_of(y_decision)

            if potential_new_state.is_legal():
                new_feats = extract_features(feature_indexer, sentence, old_state, y_decision, False, extension=extension)
                # new_feats = get_features_state(feature_cache, state_indexer, s_id, old_state, d_id)
                new_score = score_indexed_features(new_feats, weights)
                new_score += old_score

                new_counter = Counter()
                new_counter.increment_all(new_feats, -1)
                new_counter.add(old_counter)
                new_beam.add((new_counter, potential_new_state), new_score)

    return new_beam


# ======================================================
# Returns a BeamedModel trained over the given treebank.
# ======================================================
def train_beamed_model(parsed_sentences, extension, mode='SGD'):
    feature_indexer = Indexer()
    label_indexer = get_label_indexer()
    gold_decisions = []
    gold_states = []
    beam_size = 8

    # ======================================================
    # get the gold decisions and gold states
    # ======================================================
    for sentence in parsed_sentences:
        decisions, states = get_decision_sequence(sentence)
        gold_decisions.append(decisions)
        gold_states.append(states)

    # ======================================================
    # get feature cache
    # ======================================================
    feature_cache = [[[[] for k in xrange(3)] for j in xrange(len(gold_states[i]))] for i in xrange(len(parsed_sentences))]
    state_indexers = []

    for s_id, sentence in enumerate(parsed_sentences):
        if s_id % 100 == 0:
            print "Extracting features:", s_id, "/", len(parsed_sentences)

        state = initial_parser_state(len(sentence))
        curr_gold_decisions = gold_decisions[s_id]
        curr_gold_states = gold_states[s_id]
        state_indexer = Indexer()
        i = 0

        while not state.is_finished():
            gold_decision = curr_gold_decisions[i]
            d_id = label_indexer.index_of(gold_decision)
            possible_actions = get_possible_actions(state)

            for decision in possible_actions:
                d_id = label_indexer.index_of(decision)
                tmp_state = state.take_action(decision)

                if tmp_state.is_legal():
                    state_id = state_indexer.get_index(str(state))
                    feats = extract_features(feature_indexer, sentence, state, decision, True, extension=extension)
                    feature_cache[s_id][state_id][d_id] = feats
            
            next_state = state.take_action(gold_decision)
            state = next_state
            i += 1

        state_indexers.append(state_indexer)

    weights = np.zeros(shape=(len(feature_indexer),)) # np.random.rand(100000,) 
    ada = AdagradTrainer(weights, lamb=1e-7)

    # ======================================================
    # START TRAINING BEAM MODEL
    # ======================================================
    num_epochs = 5
    parsed_dev = []  

    for epoch in xrange(num_epochs):
        for s_id, sentence in enumerate(parsed_sentences):
            early_update = False

            if s_id % 100 == 0:
                print "Training:", s_id, "/", len(parsed_sentences)

            state_indexer = state_indexers[s_id]
            state = initial_parser_state(len(sentence))
            initial_features = get_features_state(feature_cache, state_indexer, s_id, state, 0) #extract_features(feature_indexer, sentence, state, 'S', True) #get_features_state(feature_cache, state_indexer, 0, state, 0)
            initial_counter = Counter()
            initial_counter.increment_all(initial_features, -1)

            beams = [Beam(beam_size)] * (2*len(sentence)+1)
            beams[0].add((initial_counter, state), score_indexed_features(initial_features, weights))

            gold_features = Counter()

            for i in xrange(2*len(sentence) + 1):
                if i < len(gold_decisions[s_id]):
                    gold_state = gold_states[s_id][i]
                    gold_id = label_indexer.index_of(gold_decisions[s_id][i])
                    gold_feats = get_features_state(feature_cache, state_indexer, s_id, gold_state, gold_id) #extract_features(feature_indexer, sentence, gold_state, gold_decision, True)
                    gold_features.increment_all(gold_feats, 1)

                if i > 0:
                    beams[i] = compute_successors(beams[i-1], feature_indexer, sentence, weights, extension)
                    # beams[i] = compute_successors(beams[i-1], feature_cache, state_indexer, s_id, weights)
                    
                curr_state_until_now = beams[i].head()[1]
                curr_score = beams[i].scores[0]
                
                gold_state_until_now = gold_states[s_id][i]
                pred_features = beams[i].head()[0]

                gradient = Counter()
                gradient.add(gold_features)
                gradient.add(pred_features)

                states_in_beam = [b[1] for b in beams[i].get_elts()]                

                # ======================================================
                # early updating
                # ======================================================
                if gold_state_until_now not in states_in_beam:
                    if mode == 'SGD':
                        for i in gradient.keys():
                            weights[i] += 0.1/(epoch+1) * gradient.get_count(i)
                    elif mode == 'ADA':
                        ada.apply_gradient_update(gradient, 1)

                    early_update = True
                    break

            # ======================================================
            # global updating
            # ======================================================
            if early_update == False:
                final_pred_features, final_pred_state = beams[-1].head()
                final_gold_state = gold_states[s_id][-1]

                # ======================================================
                # compute gradient
                # ======================================================
                if final_gold_state != final_pred_state:
                    total_gradient = Counter()
                    total_gradient.add(gold_features)
                    total_gradient.add(final_pred_features)

                    if mode == 'SGD':
                        for i in gold_features.keys():
                            weights[i] += 0.1/(epoch+1) * total_gradient.get_count(i)
                    elif mode == 'ADA':
                        ada.apply_gradient_update(total_gradient, 1)

    print "BEAM SIZE:", beam_size
    print "NUM OF EPOCHS:", num_epochs

    if mode == 'ADA':
        weights = ada.get_final_weights()
    
    return BeamedModel(feature_indexer, weights, beam_size)


# Extract features for the given decision in the given parser state. Features look at the top of the
# stack and the start of the buffer. Note that this isn't in any way a complete feature set -- play around with
# more of your own!
def extract_features(feat_indexer, sentence, parser_state, decision, add_to_indexer, extension=False):
    feats = []
    sos_tok = Token("<s>", "<S>", "<S>", "<S>")
    root_tok = Token("<root>", "<ROOT>", "<ROOT>", "<ROOT>")
    eos_tok = Token("</s>", "</S>", "</S>", "</S>")
    if parser_state.stack_len() >= 1:
        head_idx = parser_state.stack_head()
        stack_head_tok = sentence.tokens[head_idx] if head_idx != -1 else root_tok
        if parser_state.stack_len() >= 2:
            two_back_idx = parser_state.stack_two_back()
            stack_two_back_tok = sentence.tokens[two_back_idx] if two_back_idx != -1 else root_tok
        else:
            stack_two_back_tok = sos_tok
    else:
        stack_head_tok = sos_tok
        stack_two_back_tok = sos_tok
    buffer_first_tok = sentence.tokens[parser_state.get_buffer_word_idx(0)] if parser_state.buffer_len() >= 1 else eos_tok
    buffer_second_tok = sentence.tokens[parser_state.get_buffer_word_idx(1)] if parser_state.buffer_len() >= 2 else eos_tok

    # add third word
    if extension == True:
        if parser_state.stack_len() >= 3:
            three_back_idx = parser_state.stack[-3]
            stack_three_back_tok = sentence.tokens[three_back_idx] if three_back_idx != -1 else root_tok
        else:
            stack_three_back_tok = sos_tok

    buffer_third_tok = sentence.tokens[parser_state.get_buffer_word_idx(2)] if parser_state.buffer_len() >= 3 else eos_tok


    # Shortcut for adding features
    def add_feat(feat):
        maybe_add_feature(feats, feat_indexer, add_to_indexer, feat)
    add_feat(decision + ":S0Word=" + stack_head_tok.word)
    add_feat(decision + ":S0Pos=" + stack_head_tok.pos)
    add_feat(decision + ":S0CPos=" + stack_head_tok.cpos)
    add_feat(decision + ":S1Word=" + stack_two_back_tok.word)
    add_feat(decision + ":S1Pos=" + stack_two_back_tok.pos)
    add_feat(decision + ":S1CPos=" + stack_two_back_tok.cpos)

    # if extension == True: #BAD
    #     add_feat(decision + ":S2Word=" + stack_three_back_tok.word)
    #     add_feat(decision + ":S2Pos=" + stack_three_back_tok.pos)
    #     add_feat(decision + ":S2CPos=" + stack_three_back_tok.cpos)

    add_feat(decision + ":B0Word=" + buffer_first_tok.word)
    add_feat(decision + ":B0Pos=" + buffer_first_tok.pos)
    add_feat(decision + ":B0CPos=" + buffer_first_tok.cpos)
    add_feat(decision + ":B1Word=" + buffer_second_tok.word)
    add_feat(decision + ":B1Pos=" + buffer_second_tok.pos)
    add_feat(decision + ":B1CPos=" + buffer_second_tok.cpos)

    # if extension == True: #BAD
    #     add_feat(decision + ":B2Word=" + buffer_third_tok.word)
    #     add_feat(decision + ":B2Pos=" + buffer_third_tok.pos)
    #     add_feat(decision + ":B2CPos=" + buffer_third_tok.cpos)

    add_feat(decision + ":S1S0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos)
    add_feat(decision + ":S0B0Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1B0Pos=" + stack_two_back_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B1Pos=" + stack_head_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":B0B1Pos=" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":S0B0WordPos=" + stack_head_tok.word + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0PosWord=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1S0WordPos=" + stack_two_back_tok.word + "&" + stack_head_tok.pos)
    add_feat(decision + ":S1S0PosWord=" + stack_two_back_tok.pos + "&" + stack_head_tok.word)
    add_feat(decision + ":S1S0B0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0B1Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)

    if extension == True:
        # ------------- v1 extension ---------------
        add_feat(decision + ":S0WordPos=" + stack_head_tok.word + "&" + stack_head_tok.pos) #GOOD
        add_feat(decision + ":B0WordPos=" + buffer_first_tok.word + "&" + buffer_first_tok.pos) #GOOD
        add_feat(decision + ":S0Dep=" + stack_head_tok.dep) #GOOD
        add_feat(decision + ":S0PosDep=" + stack_head_tok.pos + "&" + stack_head_tok.dep) #GOOD
        add_feat(decision + ":B0Dep=" + buffer_first_tok.dep) #GOOD
        add_feat(decision + ":B0PosDep=" + buffer_first_tok.pos + "&" + buffer_first_tok.dep) #GOOD
        add_feat(decision + ":S0B0Dep=" + stack_head_tok.dep + "&" + buffer_first_tok.dep)
        add_feat(decision + ":S0S1Dep=" + stack_head_tok.dep + "&" + stack_two_back_tok.dep)
        add_feat(decision + ":S1S0B0Dep=" + stack_two_back_tok.dep + "&" + stack_head_tok.dep + "&" + buffer_first_tok.dep)
        add_feat(decision + ":S0B0Word=" + stack_head_tok.word + "&" + buffer_first_tok.word)
        add_feat(decision + ":S0S1Word=" + stack_head_tok.word + "&" + stack_two_back_tok.word)
        add_feat(decision + ":S1S0B0Word=" + stack_two_back_tok.word + "&" + stack_head_tok.word + "&" + buffer_first_tok.word)


        # add_feat(decision + ":S0WordPos=" + stack_head_tok.word + "&" + stack_head_tok.pos) #GOOD
        # add_feat(decision + ":S1WordPos=" + stack_two_back_tok.word + "&" + stack_two_back_tok.pos) #BAD
        # # add_feat(decision + ":S2WordPos=" + stack_three_back_tok.word + "&" + stack_three_back_tok.pos) #BAD

        # add_feat(decision + ":B0WordPos=" + buffer_first_tok.word + "&" + buffer_first_tok.pos) #GOOD
        # add_feat(decision + ":B1WordPos=" + buffer_second_tok.word + "&" + buffer_second_tok.pos) #BAD

        # add_feat(decision + ":S0Dep=" + stack_head_tok.dep) #GOOD
        # add_feat(decision + ":S0WordDep=" + stack_head_tok.word + "&" + stack_head_tok.dep) #GOOD
        # add_feat(decision + ":S0PosDep=" + stack_head_tok.pos + "&" + stack_head_tok.dep) #GOOD

        # add_feat(decision + ":B0Dep=" + buffer_first_tok.dep) #GOOD
        # # add_feat(decision + ":B0WordDep=" + buffer_first_tok.word + "&" + buffer_first_tok.dep) #GOOD
        # add_feat(decision + ":B0PosDep=" + buffer_first_tok.pos + "&" + buffer_first_tok.dep) #GOOD

        # add_feat(decision + ":S0B0Dep=" + stack_head_tok.dep + "&" + buffer_first_tok.dep)
        # add_feat(decision + ":S0S1Dep=" + stack_head_tok.dep + "&" + stack_two_back_tok.dep)
        # add_feat(decision + ":S1S0B0Dep=" + stack_two_back_tok.dep + "&" + stack_head_tok.dep + "&" + buffer_first_tok.dep)

        # add_feat(decision + ":B2WordPos=" + buffer_third_tok.word + "&" + buffer_third_tok.pos) #BAD

        # add_feat(decision + ":S0B0Word=" + stack_head_tok.word + "&" + buffer_first_tok.word)
        # add_feat(decision + ":S0S1Word=" + stack_head_tok.word + "&" + stack_two_back_tok.word)
        # add_feat(decision + ":S1S0B0Word=" + stack_two_back_tok.word + "&" + stack_head_tok.word + "&" + buffer_first_tok.word)
        # add_feat(decision + ":S0B0B1Word=" + stack_head_tok.word + "&" + buffer_first_tok.word + "&" + buffer_second_tok.word)

    # if extension == True: #BAD
    #     add_feat(decision + ":B0B1B2Pos=" + buffer_first_tok.pos + "&" + buffer_second_tok.pos + "&" + buffer_third_tok.pos)

    return feats


# Computes the sequence of decisions and ParserStates for a gold-standard sentence using the arc-standard
# transition framework. We use the minimum stack-depth heuristic, namely that
# Invariant: states[0] is the initial state. Applying decisions[i] to states[i] yields states[i+1].
def get_decision_sequence(parsed_sentence):
    decisions = []
    states = []
    state = initial_parser_state(len(parsed_sentence))
    while not state.is_finished():
        if not state.is_legal():
            raise Exception(repr(decisions) + " " + repr(state))
        # Look at whether left-arc or right-arc would add correct arcs
        if len(state.stack) < 2:
            result = "S"
        else:
            # Stack and buffer must both contain at least one thing
            one_back = state.stack_head()
            two_back = state.stack_two_back()
            # -1 is the ROOT symbol, so this forbids attaching the ROOT as a child of anything
            # (passing -1 as an index around causes crazy things to happen so we check explicitly)
            if two_back != -1 and parsed_sentence.get_parent_idx(two_back) == one_back and state.is_complete(two_back, parsed_sentence):
                result = "L"
            # The first condition should never be true, but doesn't hurt to check
            elif one_back != -1 and parsed_sentence.get_parent_idx(one_back) == two_back and state.is_complete(one_back, parsed_sentence):
                result = "R"
            elif len(state.buffer) > 0:
                result = "S"
            else:
                result = "R" # something went wrong, buffer is empty, just do right arcs to finish the tree
        decisions.append(result)
        states.append(state)
        if result == "L":
            state = state.left_arc()
        elif result == "R":
            state = state.right_arc()
        else:
            state = state.shift()
    states.append(state)
    return (decisions, states)
