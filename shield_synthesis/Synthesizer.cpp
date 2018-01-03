#include "Synthesizer.h"
#include <math.h>
#include <assert.h>

Synthesizer::Synthesizer(ShieldMonitor* s_monitor, int num_inputs, int num_design_choices) :
num_inputs_(num_inputs),
s_monitor_(s_monitor)
{
    initial_state_bdd_ = One();
    transition_bdd_ = One();
    num_design_choices_ = num_design_choices;
    
    synthesize();
}

void Synthesizer::synthesize()
{
    int variable_idx = 0;
    for (int idx = 1; idx <= s_monitor_->num_inputs_; idx++) {
        if (input_vars_.find(s_monitor_->names_[idx]) == input_vars_.end()) {
            input_vars_.insert(s_monitor_->names_[idx]);
            variables_[s_monitor_->names_[idx]] = variable_idx;
            variable_idx++;
        }
    }
    for (int idx = s_monitor_->num_inputs_ + 1; idx <= s_monitor_->num_inputs_ + s_monitor_->num_outputs_; idx++) {
        if (output_vars_.find(s_monitor_->names_[idx]) == output_vars_.end()) {
            output_vars_.insert(s_monitor_->names_[idx]);
            variables_[s_monitor_->names_[idx]] = variable_idx;
            variable_idx++;
        }
    }
    
    // std::cout << "Input variables: \n";
//     for (auto name : input_vars_) {
//         std::cout << name << std::endl;
//     }
//
//     std::cout << "Output variables: \n";
//     for (auto name : output_vars_) {
//         std::cout << name << std::endl;
//     }
    
    encode_states();
    encode_variables();    
    
    for (int state_pos = 0; state_pos < num_of_bits_; state_pos++) {
        next_state_vars_bdds_.push_back(bdds_["s" + std::to_string(num_of_bits_ - 1 - state_pos) + "n"]);
    }
    
    for (auto name : input_vars_) {
        in_vars_bdds_.push_back(bdds_["v" + std::to_string(variables_[name])]);
    }
    
    for (auto name : output_vars_) {
        out_vars_bdds_.push_back(bdds_["v" + std::to_string(variables_[name])]);
    }
        
    encode_transitions();
    // Cudd_PrintDebug(cudd_.getManager(), transition_bdd_.getNode(), 4, 10000);
    
    encode_error_states();
    
    win_region_ = encode_winning_region();
    auto strategy = nondet_admissible_strategy();
    
    auto strategy_multiple = strategy_for_mulitple_desgin_choices(strategy, num_design_choices_);
    
    func_by_var_ = extract_output_funcs(strategy_multiple);
}

void Synthesizer::encode_states() 
{
    num_of_bits_ = ceil(log2(s_monitor_->nodes_.size()));;
    
    for (int state_pos = 0; state_pos < num_of_bits_; state_pos++) {
        auto node_bdd = IthVar(bdds_.size());
        std::string state_name = "s" + std::to_string(num_of_bits_ - 1 - state_pos);
        bdd_names_.push_back(state_name);
        bdds_[state_name] = node_bdd;            
    }
    
    for (int state_pos = 0; state_pos < num_of_bits_; state_pos++) {
        auto node_bdd = IthVar(bdds_.size());
        std::string state_name = "s" + std::to_string(num_of_bits_ - 1 - state_pos) + "n";
        bdd_names_.push_back(state_name);
        bdds_[state_name] = node_bdd;
    }
    
    initial_state_bdd_ = encode_node(s_monitor_->initial_nodes()[0]);
}

BDD Synthesizer::encode_node(Dfa::Node* node, bool next)
{
    int num_bits = ceil(log2(s_monitor_->nodes_.size()));
    
    BDD f = One();
    for (int i = 0; i < num_bits; i++) {
        std::string name = "s" + std::to_string(i);
        if (next) name += "n";
        if ((((node->id_ - 1) >> i) & 1) == 1) {
            f &= bdds_[name];
        } else {
            f &= ~bdds_[name];
        }
    }
    
    return f;
}

void Synthesizer::encode_variables()
{
    for (auto pair : variables_) {
        bdd_names_.push_back(pair.first);
        auto node_bdd = IthVar(bdds_.size());
        bdds_["v" + std::to_string(pair.second)] = node_bdd;
    }
}

void Synthesizer::encode_transitions()
{    
    auto f = Zero();
    
    for (auto node : s_monitor_->nodes_) {
        BDD source = encode_node(node);
        
        for (auto edge : node->edges_) {
            BDD target = encode_node(edge->target_, true);
            BDD transition = source & target;
            
            for (label_size_t mask = 1, idx = 1; mask < ((label_size_t)1) << (s_monitor_->num_inputs_ + s_monitor_->num_outputs_); mask <<= 1, idx++) {
                if ((mask & edge->label_.mask) == mask) {
                    auto var_bdd = bdds_["v" + std::to_string(variables_[s_monitor_->names_[idx]])];
                    if ((mask & edge->label_.sign) == mask) {
                        transition &= var_bdd;
                    } else {
                        transition &= ~var_bdd;
                    }
                }
            }
            
            f += transition;
        }
    }
    
    transition_bdd_ = f;
}

void Synthesizer::encode_error_states()
{    
    error_state_bdd_ = Zero();
    for (auto node : s_monitor_->final_nodes()) {
        error_state_bdd_ += encode_node(node);
        // std::cout << "node: " << node->to_string() << " | "<< error_state_bdd_ << "\n";
        
    }
}

BDD Synthesizer::encode_winning_region() 
{
    auto not_error_bdd = ~error_state_bdd_;
    auto new_set_bdd = One();
    while (true) {
        auto current_set_bdd = new_set_bdd;
        new_set_bdd = not_error_bdd & pre_sys_bdd(current_set_bdd, transition_bdd_);
        
        if ((new_set_bdd & initial_state_bdd_) == Zero()) {
            return Zero();
        }
        
        if (new_set_bdd == current_set_bdd) {
            return new_set_bdd;
        }
    }
}

BDD Synthesizer::encode_buchi_winning_region(BDD state_space, BDD acc_state_bdd) 
{
    auto new_set_bdd_gfp = state_space;
    while (true) {
        auto current_set_bdd_gfp = new_set_bdd_gfp;
        
        auto new_set_bdd_lfp = Zero();
        while(true) {
            auto current_set_bdd_lfp = new_set_bdd_lfp;
            new_set_bdd_lfp = ((acc_state_bdd & current_set_bdd_gfp & pre_bdd(current_set_bdd_gfp, transition_bdd_)) | 
                                current_set_bdd_lfp | pre_bdd(current_set_bdd_lfp, transition_bdd_)) & state_space;
            if (new_set_bdd_lfp == current_set_bdd_lfp) {
                break;
            }
            attractor_bdds_.push_back(new_set_bdd_lfp);
        }
        
        new_set_bdd_gfp = new_set_bdd_lfp;
        
        if((new_set_bdd_gfp & initial_state_bdd_) == Zero()) {
            return Zero();
        }
        
        if (new_set_bdd_gfp == current_set_bdd_gfp) {
            return new_set_bdd_gfp;
        }
    }
}

BDD Synthesizer::encode_acceptance_states() 
{
    auto acceptance_state_bdd = Zero();
    for (auto node : s_monitor_->nodes_) {
        ShieldMonitor::ShieldMonitorNode* s_node = (ShieldMonitor:: ShieldMonitorNode*) node;
        if (s_node->design_error_ == 0) {
            acceptance_state_bdd += encode_node(s_node);
        }
    }
    
    return acceptance_state_bdd;
}

BDD Synthesizer::pre_bdd(BDD dst_states_bdd, BDD transition_bdd) 
{
    auto primed_dst_states_bdd = prime_states(dst_states_bdd);
    auto intersection = transition_bdd & primed_dst_states_bdd;
    auto exists_outs = intersection;
    if (out_vars_bdds_.size() > 0) {
        exists_outs = intersection.ExistAbstract(get_cube(out_vars_bdds_));
    }
    
    auto next_state_vars_cube = prime_states(get_cube(get_all_state_bdds()));
    auto exist_next_state = exists_outs.ExistAbstract(next_state_vars_cube);
    
    auto exists_inputs = exist_next_state;
    if (in_vars_bdds_.size() > 0) {
        exists_inputs = exist_next_state.ExistAbstract(get_cube(in_vars_bdds_));
    }
    
    return exists_inputs;
}

BDD Synthesizer::pre_sys_bdd(BDD dst_states_bdd, BDD transition_bdd) 
{
    auto primed_dst_states_bdd = prime_states(dst_states_bdd);
    auto intersection = transition_bdd & primed_dst_states_bdd;
    
    auto exists_outs = intersection;
    if (out_vars_bdds_.size() > 0) {
        auto out_vars_cube_bdd = get_cube(out_vars_bdds_);        
        exists_outs = intersection.ExistAbstract(out_vars_cube_bdd);
    }
    
    auto next_state_vars_cube = prime_states(get_cube(get_all_state_bdds()));
    auto exist_next_state = exists_outs.ExistAbstract(next_state_vars_cube);
    
    auto forall_inputs = exist_next_state;
    if (in_vars_bdds_.size() > 0) {
        auto in_vars_cube_bdd = get_cube(in_vars_bdds_);        
        forall_inputs = exist_next_state.UnivAbstract(in_vars_cube_bdd);
    }
    
    return forall_inputs;
}

BDD Synthesizer::prime_states(BDD state_bdd) 
{
    std::vector<BDD> primed_var_array;
    std::vector<BDD> curr_var_array;
    for (auto pair : bdds_) {
        curr_var_array.push_back(pair.second);
        int index = pair.second.NodeReadIndex();
        
        BDD new_l_bdd = pair.second;
        if (index >= 0 && index < num_of_bits_) {
            new_l_bdd = IthVar(index + num_of_bits_);
        }
        
        primed_var_array.push_back(new_l_bdd);
    }
    
    auto replaced_states_bdd = state_bdd.SwapVariables(curr_var_array, primed_var_array);
    
    return replaced_states_bdd;
}

template<class TContainer>
BDD Synthesizer::get_cube(TContainer variables)
{
    auto cube = One();
    for (auto var : variables) {
        cube &= var;
    }
    return cube;
}

std::vector<BDD> Synthesizer::get_all_state_bdds()
{
    std::vector<BDD> states;
    for (int i = 0; i < num_of_bits_; i++) {
        states.push_back(bdds_["s" + std::to_string(i)]);
    }
    return states;
}

BDD Synthesizer::nondet_admissible_strategy() 
{    
    auto acc_bdd = encode_acceptance_states();

    encode_buchi_winning_region(win_region_, acc_bdd);

    assert(!attractor_bdds_.empty());
    assert(win_region_ == attractor_bdds_.back());
    for (int i = 1; i < attractor_bdds_.size(); i++) {
        assert((attractor_bdds_[i - 1] & ~attractor_bdds_[i]) == Zero());
    }

    auto strategy_1 = Zero();
    auto strategy_2 = Zero();

    auto primed_attractor = prime_states(attractor_bdds_[0]);
    strategy_1 += attractor_bdds_[0] & transition_bdd_ & primed_attractor;

    for (int i = 1; i < attractor_bdds_.size(); i++) {
        auto attractor_diff = attractor_bdds_[i] & ~attractor_bdds_[i - 1];
        primed_attractor = prime_states(attractor_bdds_[i - 1]);
        strategy_1 += attractor_diff & transition_bdd_ & primed_attractor;
    }

    auto primed_last_attractor = prime_states(attractor_bdds_.back());
    auto exists_eliminated_strategy = strategy_1;
    if (out_vars_bdds_.size() > 0) {
        exists_eliminated_strategy = strategy_1.ExistAbstract(get_cube(out_vars_bdds_));
    }

    auto next_state_vars_cube = prime_states(get_cube(get_all_state_bdds()));
    exists_eliminated_strategy = exists_eliminated_strategy.ExistAbstract(next_state_vars_cube);

    strategy_2 = attractor_bdds_[0] & transition_bdd_ & primed_last_attractor & ~exists_eliminated_strategy;
    for (int i = 1; i < attractor_bdds_.size(); i++) {
        auto attractor_diff = attractor_bdds_[i] & ~attractor_bdds_[i - 1];
        strategy_2 += attractor_diff & transition_bdd_ & primed_last_attractor & ~exists_eliminated_strategy;
    }

    auto strategy = strategy_1 | strategy_2;
    strategy = strategy.ExistAbstract(next_state_vars_cube);
    
    // auto primed_win_region = prime_states(win_region_);
 //    auto intersection = primed_win_region & transition_bdd_ & win_region_;
 //
 //    auto next_state_vars_cube = prime_states(get_cube(get_all_state_bdds()));
 //    auto strategy = intersection.ExistAbstract(next_state_vars_cube);
        
    return strategy;
}

std::map<std::string, BDD> Synthesizer::extract_output_funcs(BDD strategy) 
{
    std::map<std::string, BDD> output_models;
     
    for (std::string c_name : output_vars_) {
        auto c = bdds_["v" + std::to_string(variables_[c_name])];
        
        std::vector<BDD> others;
        for (auto bdd : out_vars_bdds_) {
            if (bdd != c) {
                others.push_back(bdd);
            }
        }

        auto c_arena = strategy;
        if (others.size() > 0) {
            c_arena = strategy.ExistAbstract(get_cube(others));
        }

        auto can_be_true = c_arena.Cofactor(c);
        auto can_be_false = c_arena.Cofactor(~c);

        auto must_be_true = (~can_be_false) & can_be_true;
        auto must_be_false = (~can_be_true) & can_be_false;

        auto care_set = must_be_true | must_be_false;

        auto c_model = must_be_true.Restrict(care_set);

        output_models[c_name] = c_model;

        strategy &= (c & c_model) | (~c & ~c_model);
    }
    
    strategy &= transition_bdd_;

    for (int i = 0; i < num_of_bits_; i++) {
        std::string state_name = "s" + std::to_string(i) + "n";
        auto c = bdds_[state_name];
        
        std::vector<BDD> others;
        others.insert(others.end(), out_vars_bdds_.begin(), out_vars_bdds_.end());
        for (auto bdd : next_state_vars_bdds_) {
            if (bdd != c) {
                others.push_back(bdd);
            }
        }
      
        auto c_arena = strategy;
        if (others.size() > 0) {
            c_arena = strategy.ExistAbstract(get_cube(others));
        }

        auto can_be_true = c_arena.Cofactor(c);
        auto can_be_false = c_arena.Cofactor(~c);

        auto must_be_true = (~can_be_false) & can_be_true;
        auto must_be_false = (~can_be_true) & can_be_false;

        auto care_set = must_be_true | must_be_false;

        auto c_model = must_be_true.Restrict(care_set);

        output_models[state_name] = c_model;

        strategy &= (c & c_model) | (~c & ~c_model);
    }
    
    return output_models;
}

std::string Synthesizer::walk(DdNode* bdd, OutputFormatter* formatter)
{
    if (Cudd_IsConstant(bdd)) {
        return formatter->constant(bdd == Zero().getNode() ? false : true);
    }
    
    if (visited_.find(bdd) != visited_.end()) {
        return visited_[bdd];
    }
    
    auto node_name = "tmp" + std::to_string(tmp_count_++);
    formatter->add_temporary(node_name);
    
    visited_[bdd] = node_name;
    bdd_node_counter_++;
    
    auto node_idx = Cudd_NodeReadIndex(bdd);
    auto cond = bdd_names_[node_idx];
    
    auto t_lit = walk(Cudd_T(bdd), formatter);
    auto e_lit = walk(Cudd_E(bdd), formatter);
    
    auto res = formatter->multiplexer(cond, t_lit, e_lit);
    if (Cudd_IsComplement(bdd)) {
        res = formatter->negate(res);
    }
    
    formatter->assign(node_name, res);
    
    return node_name;
}

void Synthesizer::model_to_output_format(std::string c_name, BDD c_bdd, BDD func_bdd, OutputFormatter* formatter)
{
    visited_.clear();
    bdd_node_counter_ = 1;
    
    auto top_level_var = walk(func_bdd.getNode(), formatter);
    formatter->assign(c_name, top_level_var);
}

void Synthesizer::to_output_format(OutputFormatter* formatter)
{
    tmp_count_ = 1;
    
    if (func_by_var_.empty()) {
        std::cout << "Synthesis was not successful\n";
    }
    if (win_region_ == Zero()) {
        std::cout << "Winning region is empty\n";
    }
    
    for (int i = 1; i <= num_inputs_; i++) {
        formatter->add_input(s_monitor_->names_[i]);
    }
    
    for (int counter = 1; counter <= num_design_choices_; counter++) {
        for (int i = num_inputs_ + 1; i <= s_monitor_->num_inputs_; i++) {
            formatter->add_input(s_monitor_->names_[i] + "_" + std::to_string(counter));
        }
    }
    
    
    for (int i = s_monitor_->num_inputs_ + 1; i <= s_monitor_->num_inputs_ + s_monitor_->num_outputs_; i++) {
        std::string name = s_monitor_->names_[i];
        formatter->add_output(name);
        auto var_bdd = bdds_["v" + std::to_string(variables_[name])];
        auto node_idx = var_bdd.NodeReadIndex();
        auto var_name = bdd_names_[node_idx];
        model_to_output_format(var_name, var_bdd, func_by_var_[name], formatter);
    }
    
    for (int state_pos = 0; state_pos < num_of_bits_; state_pos++) {
        formatter->add_variable("s" + std::to_string(num_of_bits_ - 1 - state_pos));
        auto state_name = "s" + std::to_string(num_of_bits_ - 1 - state_pos) + "n";
        auto state_bdd = bdds_[state_name];
        model_to_output_format(state_name, state_bdd, func_by_var_[state_name], formatter);
    }
}

BDD Synthesizer::strategy_for_mulitple_desgin_choices(BDD strategy, int num_design_choices)
{
    BDD first_choice;
    BDD transition_bdd_multiple_;
    std::vector<BDD> abstract_outputs;
    std::vector<BDD> shield_outputs;
    
    for (int i = num_inputs_ + 1; i <= s_monitor_->num_inputs_; i++) {
        abstract_outputs.push_back(bdds_["v" + std::to_string(variables_[s_monitor_->names_[i]])]);
        shield_outputs.push_back(bdds_["v" + std::to_string(variables_[s_monitor_->names_[s_monitor_->num_inputs_ - num_inputs_ + i]])]);
    }
    
    std::vector<BDD> sub_strategies;
    std::vector<BDD> sub_transitions;
    
    BDD new_strategy = Zero();
    BDD new_transitions = Zero();
    
    for (int i = 0; i < num_design_choices; i++) {
        std::vector<BDD> concrete_outputs;
        
        
        for (auto old_bdd : abstract_outputs) {
            BDD new_bdd = IthVar(bdds_.size());
            std::string state_name = bdd_names_[old_bdd.NodeReadIndex()] + "_" + std::to_string(i + 1);
            std::cout << state_name << ": " << new_bdd << std::endl;
            bdd_names_.push_back(state_name);
            bdds_[state_name] = new_bdd;
            concrete_outputs.push_back(new_bdd);
        }
         
        auto substituted = strategy.SwapVariables(abstract_outputs, concrete_outputs);
        if (i == 0) {
            first_choice = substituted;
        }
        
        substituted &= make_vectors_equal(shield_outputs, concrete_outputs);  
        
        sub_transitions.push_back(transition_bdd_.SwapVariables(abstract_outputs, concrete_outputs));
        
        sub_strategies.push_back(substituted); 
        
    }
    
    // strategies for provided design output possibilities
    BDD forbidden_strategies = One();
    for (int i = 0; i < sub_strategies.size(); i++) {
        auto sub_strategy = sub_strategies[i];
        auto sub_transition = sub_transitions[i];        
        new_strategy += forbidden_strategies.UnivAbstract(get_cube(shield_outputs)) & sub_strategy;
        new_transitions += forbidden_strategies.UnivAbstract(get_cube(shield_outputs)) & sub_strategy & sub_transition;
        
        forbidden_strategies &= ~sub_strategy;
    }
    
    // strategy in case all provided design outputs are invalid 
    new_strategy += forbidden_strategies.UnivAbstract(get_cube(shield_outputs)) & first_choice;
    new_transitions += forbidden_strategies.UnivAbstract(get_cube(shield_outputs)) & sub_transitions[0];        
    
    BDD test = new_strategy.ExistAbstract(get_cube(abstract_outputs));
    assert(test == new_strategy);
    
    transition_bdd_ = new_transitions;
    
    return new_strategy;
}

BDD Synthesizer::make_vectors_equal(std::vector<BDD> first, std::vector<BDD> second) 
{
    assert(first.size() == second.size());
    
    BDD res = One();
    for (int i = 0; i < first.size(); i++) {
        res &= ~(first[i] ^ second[i]);
    }
    
    return res;
}
