# models.py

from utils import *
from adagrad_trainer import *
from treedata import *
import numpy as np
import random

ROOT = -1

# Greedy parsing model. This model treats shift/reduce decisions as a multiclass classification problem.
class GreedyModel(object):
    def __init__(self, feature_indexer, feature_weights):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        fi = self.feature_indexer
        weights = self.feature_weights
        # print weights
        state = initial_parser_state(len(sentence))
        decisions = []
        states = [state]

        label_indexer = get_label_indexer()

        while not state.is_finished():
            # raw_input("parse")

            argmax = [np.NINF, np.NINF, np.NINF]
            possible_actions = get_possible_actions(state)
            denominator = np.NINF

            # calculate denominator
            for decision in possible_actions:
                y_prime = label_indexer.index_of(decision)
                feat = extract_features(fi, sentence, state, decision, False)
                tmp = score_indexed_features(feat, weights)
                new_state = state.take_action(decision)

                # print "decision", decision, "feat", feat

                if new_state.is_legal():
                    denominator = np.logaddexp(denominator, tmp)

            # print "denominator", denominator

            # calculate numerator
            for decision in possible_actions:
                d_id = label_indexer.index_of(decision)
                new_state = state.take_action(decision)

                if new_state.is_legal():
                    feat = extract_features(fi, sentence, state, decision, False)
                    numerator = score_indexed_features(feat, weights)
                    prob = numerator - denominator
                    argmax[d_id] = prob

            # print "numerator", numerator, "feat", feat

            decision = label_indexer.get_object(np.argmax(argmax))
            prob = np.amax(argmax)

            # print [np.exp(a) for a in argmax], decision, state

            prob = np.exp(prob)
            new_state = state.take_action(decision)
            states.append(new_state)
            decisions.append(decision)
            state = new_state

        dependencies = states[-1].get_dep_objs(len(sentence))
        return ParsedSentence(sentence.tokens, dependencies)



# Beam-search-based global parsing model. Shift/reduce decisions are still modeled with local features, but scores are
# accumulated over the whole sequence of decisions to give a "global" decision.
class BeamedModel(object):
    def __init__(self, feature_indexer, feature_weights, beam_size=1):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.beam_size = beam_size
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        raise Exception("IMPLEMENT ME")


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


def get_features(feature_cache, s_id, state_id, d_id):
    key = (s_id, state_id, d_id)
    if key in feature_cache.keys():
        return feature_cache[key]
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


# Returns a GreedyModel trained over the given treebank.
def train_greedy_model(parsed_sentences):
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
    sentence_states = {}
    feature_cache = {}

    for s_id, sentence in enumerate(parsed_sentences):
        if s_id % 100 == 0:
            print "Extracting features:", s_id, "/", len(parsed_sentences)

        state = initial_parser_state(len(sentence))
        state_indexer = Indexer()
        curr_gold_decisions = gold_decisions[s_id]
        curr_gold_states = gold_states[s_id]
        i = 0

        while not state.is_finished():
            # raw_input('state')
            gold_decision = curr_gold_decisions[i]
            d_id = label_indexer.index_of(gold_decision)
            state_id = state_indexer.get_index((str(state), gold_decision))

            key = (s_id, state_id, d_id)
            feats = extract_features(feature_indexer, sentence, state, gold_decision, True)
            feature_cache[key] = feats

            # print state
            # print curr_gold_states[i]
            
            next_state = state.take_action(gold_decision)
            state = next_state
            i += 1

        sentence_states[s_id] = state_indexer

    weights = np.ones(shape=(len(feature_indexer),))
    # return

    # ======================================================
    # build a classifier (logistic regression)
    # ======================================================
    parsed_dev = []
    num_epochs = 1

    for epoch in xrange(num_epochs):
        for s_id, sentence in enumerate(parsed_sentences):
            if s_id % 100 == 0:
                print "Training:", s_id, "/", len(parsed_sentences)

            state = initial_parser_state(len(sentence))
            decisions = []
            states = [state]
            sentence_gold_states = gold_states[s_id]
            sentence_gold_decisions = gold_decisions[s_id] 
            state_indexer = sentence_states[s_id]
            state_id = state_indexer.index_of((str(state), 'S'))
            index = 0

            while not state.is_finished():
                # first decision should be shift
                # calculate argmax of S, L, R
                argmax = [np.NINF, np.NINF, np.NINF]
                possible_actions = get_possible_actions(state)
                denominator = np.NINF

                # calculate numerator and denominator
                for i, y_decision in enumerate(possible_actions):
                    numerator = np.NINF
                    d_id = label_indexer.index_of(y_decision)
                    next_state = state.take_action(y_decision)

                    # calculate numerator functions
                    if next_state.is_legal():
                        state_id = state_indexer.index_of((str(state), y_decision))
                        # print "State ID", state_id, state, y_decision
                        # print next_state
                        feats = get_features(feature_cache, s_id, state_id, d_id)
                        # calculate probability
                        # P(y|x) = exp(wT . f(x,y)) / sum(exp(wT . f(x,y'))) over y' in decisions
                        numerator = score_indexed_features(feats, weights)
                        denominator = numerator
                        # calculate denominator
                        popped = list(possible_actions)
                        popped.pop(i)
                        # print possible_actions, popped

                        for j, y_prime in enumerate(popped):
                            state_id2 = state_indexer.index_of((str(state), y_prime))
                            d_id2 = label_indexer.index_of(y_prime)
                            feats2 = get_features(feature_cache, s_id, state_id2, d_id2)
                            tmp = score_indexed_features(feats2, weights)
                            tmp_state = state.take_action(y_prime)
                            
                            if tmp_state.is_legal():
                                denominator = np.logaddexp(denominator, tmp)          

                        prob = numerator - denominator
                        # print state_id, prob, numerator, denominator
                        argmax[d_id] = prob

                # print [np.exp(a) for a in argmax]
                decision = label_indexer.get_object(np.argmax(argmax))
                decisions.append(decision)
                next_state = state.take_action(decision)
                states.append(next_state)

                
                # ======================================================
                # compute log likelihood
                # compute gradient
                # ======================================================
                # gold features
                gold_features = Counter()
                # expected_features = Counter()
                gradient = Counter()
                # prob_features = Counter()
                state_id = state_indexer.index_of((str(state), sentence_gold_decisions[index]))

                # print index, state_id
                # print "state", state
                # print "state indexer", state_indexer.get_object(state_id)
                # print "gold decision", sentence_gold_decisions[index]
                # print "predicted decision", decision
                
                y_star_id = label_indexer.index_of(sentence_gold_decisions[index])
                gold_feats = get_features(feature_cache, s_id, index, y_star_id)
                gold_tmp = score_indexed_features(gold_feats, weights)
                gold_features.increment_all(gold_feats, gold_tmp)
                
                # predicted S
                pred_s = Counter()
                prob_s = Counter()
                s_feats = get_features(feature_cache, s_id, index, 0)
                s_tmp = score_indexed_features(s_feats, weights)
                prob_s.increment_all(s_feats, -np.exp(argmax[0]))
                pred_s.increment_all(s_feats, np.exp(s_tmp))

                # predicted L
                pred_l = Counter()
                prob_l = Counter()
                l_feats = get_features(feature_cache, s_id, index, 1)
                l_tmp = score_indexed_features(l_feats, weights)
                prob_l.increment_all(l_feats, -np.exp(argmax[1]))
                pred_l.increment_all(l_feats, np.exp(l_tmp))

                # predicted R
                pred_r = Counter()
                prob_r = Counter()
                r_feats = get_features(feature_cache, s_id, index, 2)
                r_tmp = score_indexed_features(r_feats, weights)
                prob_r.increment_all(r_feats, -np.exp(argmax[2]))
                pred_r.increment_all(r_feats, np.exp(r_tmp))

                pred_s.add(pred_l)
                pred_s.add(pred_r)

                prob_s.add(prob_l)
                prob_s.add(prob_r)

                # print pred_s
                for key in pred_s.keys():
                    value = pred_s.get_count(key)
                    pred_s.set_count(key, -np.log(value))

                # print "pred s", pred_s
                gold_features.add(pred_s)

                # print "prob s", prob_s
                gradient.increment_all(gold_feats, 1)
                gradient.add(prob_s)

                # print "gold feature", gold_features
                # print "gradient", gradient

                # raw_input('bla')

                for i in gradient.keys():
                    weights[i] += 0.1 * gradient.get_count(i)
                
                index += 1
                state = next_state

        # # end while            
        # parsed_dev.append(ParsedSentence(sentence.tokens, states[-1].get_dep_objs(len(sentence))))

    # print_evaluation(parsed_sentences, parsed_dev)

    return GreedyModel(feature_indexer, weights)

# Returns a BeamedModel trained over the given treebank.
def train_beamed_model(parsed_sentences):
    raise Exception("IMPLEMENT ME")


# Extract features for the given decision in the given parser state. Features look at the top of the
# stack and the start of the buffer. Note that this isn't in any way a complete feature set -- play around with
# more of your own!
def extract_features(feat_indexer, sentence, parser_state, decision, add_to_indexer):
    feats = []
    sos_tok = Token("<s>", "<S>", "<S>")
    root_tok = Token("<root>", "<ROOT>", "<ROOT>")
    eos_tok = Token("</s>", "</S>", "</S>")
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
    # Shortcut for adding features
    def add_feat(feat):
        maybe_add_feature(feats, feat_indexer, add_to_indexer, feat)
    add_feat(decision + ":S0Word=" + stack_head_tok.word)
    add_feat(decision + ":S0Pos=" + stack_head_tok.pos)
    add_feat(decision + ":S0CPos=" + stack_head_tok.cpos)
    add_feat(decision + ":S1Word=" + stack_two_back_tok.word)
    add_feat(decision + ":S1Pos=" + stack_two_back_tok.pos)
    add_feat(decision + ":S1CPos=" + stack_two_back_tok.cpos)
    add_feat(decision + ":B0Word=" + buffer_first_tok.word)
    add_feat(decision + ":B0Pos=" + buffer_first_tok.pos)
    add_feat(decision + ":B0CPos=" + buffer_first_tok.cpos)
    add_feat(decision + ":B1Word=" + buffer_second_tok.word)
    add_feat(decision + ":B1Pos=" + buffer_second_tok.pos)
    add_feat(decision + ":B1CPos=" + buffer_second_tok.cpos)
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
