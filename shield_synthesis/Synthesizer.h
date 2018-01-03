#include "ShieldMonitor.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuddObj.hh"
#include <map>
#include <vector>
#include "OutputFormatter.h"
#include <sstream>

class Synthesizer 
{
    Cudd cudd_;
    BDD one_;
    BDD zero_;
    int num_of_bits_;
    int num_inputs_;
    ShieldMonitor* s_monitor_;
    std::set<std::string> input_vars_;
    std::set<std::string> output_vars_;
    std::map<Dfa*, int> state_offsets_;
    std::map<std::string, int> variables_;
    std::vector<std::string> bdd_names_;
    std::map<std::string, BDD> bdds_;
    BDD initial_state_bdd_;
    BDD transition_bdd_;
    BDD error_state_bdd_;
    BDD win_region_;
    std::map<std::string, BDD> func_by_var_;
    std::vector<BDD> next_state_vars_bdds_;
    std::vector<BDD> in_vars_bdds_;
    std::vector<BDD> out_vars_bdds_;
    std::vector<BDD> attractor_bdds_;
    std::map<DdNode*, std::string> visited_;
    int tmp_count_;
    int bdd_node_counter_;
    int num_design_choices_;
    
public:
    Synthesizer(ShieldMonitor* s_monitor, int num_inputs, int num_design_choices = 1);
    void synthesize();
    void encode_states(); 
    BDD encode_node(Dfa::Node* node, bool next = false);
    void encode_variables();
    void encode_transitions();
    void encode_error_states();
    BDD encode_winning_region();
    BDD pre_sys_bdd(BDD dst_states_bdd, BDD transition_bdd);
    BDD prime_states(BDD state_bdd);
    template<class TContainer> BDD get_cube(TContainer variables);
    std::vector<BDD> get_all_state_bdds();
    BDD nondet_admissible_strategy();
    BDD encode_buchi_winning_region(BDD state_space, BDD acc_state_bdd);
    BDD encode_acceptance_states();
    BDD pre_bdd(BDD dst_states_bdd, BDD transition_bdd);
    std::map<std::string, BDD> extract_output_funcs(BDD strategy);
    std::string walk(DdNode* bdd, OutputFormatter* formatter);
    void to_output_format(OutputFormatter* formatter);
    void model_to_output_format(std::string c_name, BDD c_bdd, BDD func_bdd, OutputFormatter* formatter);
    BDD strategy_for_mulitple_desgin_choices(BDD strategy, int num_design_choices);
    BDD make_vectors_equal(std::vector<BDD> first, std::vector<BDD> second);
    
    inline BDD One() {
        return cudd_.bddOne();
    }
    
    inline BDD Zero() {
        return cudd_.bddZero();
    } 
    
    inline BDD IthVar(int index) {
        return cudd_.bddVar(index);
    }
};
