#include "ShieldMonitor.h"
#include <list>
#include <assert.h>
#include <set>
#include <map>
#include <iostream>
#include <sstream>

ShieldMonitor::ShieldMonitor(Dfa* spec) :
Dfa(0, spec->num_inputs_ + spec->num_outputs_, spec->num_outputs_ + 1)
{
    for (int i = 1; i <= spec->num_inputs_; i++) {
        names_[i] = spec->names_[i];
    }
    
    for (int i = spec->num_inputs_ + 1; i <= spec->num_inputs_ + spec->num_outputs_; i++) {
        names_[i] = spec->names_[i];
        names_[i + spec->num_outputs_] = spec->names_[i] + "__s";
    }
    
    names_[num_inputs_ + num_outputs_] = "recovery__s";
    
    std::vector<Node*> good_nodes;
    for (auto node : spec->nodes_) {
        if (!node->final_) good_nodes.push_back(node);
    }
    
    for (auto node : good_nodes) {
        Node* original = add_node(new ShieldMonitorNode(node->id_)); // original states (everything fine til here)
        if (node->initial_) original->initial_ = true;
        add_node(new ShieldMonitorNode(node->id_ + spec->nodes_.size(), 1)); // copies of the original states (design error occured)
    }
    
    Node* final;
    for (auto node : spec->final_nodes()) {
        Node* added = add_node(new ShieldMonitorNode(node->id_)); // bad states are not duplicated
        final = added;
        added->final_ = true;
        
        for (auto edge : node->edges_) {
            add_edge(added, added, edge->label_);
        }
    }
  
    for (auto node : good_nodes) {
        auto source = find_node_by_id(node->id_);
        for (auto edge : node->edges_) {
            auto target = find_node_by_id(edge->target_->id_);
            if (target->final_) {
                // create a label with x__s set to x and inputs are the same
                label_size_t mask = ~(((label_size_t) 1 << spec->num_inputs_) - 1);
                label_t label = edge->label_.shift(spec->num_outputs_, mask);

                add_edge(source, final, label);
                add_edge(find_node_by_id(source->id_ + spec->nodes_.size()), final, label);

                // add edges to the recovery zone
                for (auto other_edge : node->edges_) {
                    if (!other_edge->target_->final_ && spec->check_label_compatibility(edge->label_, other_edge->label_)) {
                        label = edge->label_ | other_edge->label_.shift(spec->num_outputs_, mask);
                        add_edge(source, find_node_by_id(other_edge->target_->id_ + spec->nodes_.size()), label);
                    }
                }
            } else {
                // create a mask having all outputs bits set and the input bits like in the original mask
                label_size_t input_mask = ~(((label_size_t) 1 << spec->num_inputs_) - 1);
                label_size_t mask = ~((((label_size_t) 1) << (num_inputs_ + num_outputs_ - 1)) - 1);
                mask ^= input_mask;
                mask |= edge->label_.mask;

                label_t label;
                label.mask = mask;
                // add normal edges but guarantee no shield deviation
                for (label_size_t sign = 0; sign < (((label_size_t)1) << spec->num_outputs_); sign++) {
                    label.sign = ((sign << spec->num_outputs_) | sign) << spec->num_inputs_;

                    if (((label.sign ^ edge->label_.sign) & edge->label_.mask & input_mask) == 0) {
                        label.sign |= (edge->label_.sign & ~input_mask);
                        add_edge(source, target, label);
                    }
                }

                // add bad edges: shield deviation even if design was correct
                for (label_size_t mask = 1 << spec->num_inputs_; mask < (((label_size_t)1) << (spec->num_inputs_ + spec->num_outputs_)); mask <<= 1) {
                    label.mask = (mask << spec->num_outputs_) | mask;
                    label.sign = mask;
                    if (((label.sign ^ edge->label_.sign) & mask & edge->label_.mask) == 0) {
                        label = label | edge->label_;
                        add_edge(source, final, label);
                    }

                    label.mask = (mask << spec->num_outputs_) | mask;
                    label.sign = (mask << spec->num_outputs_);
                    if (((label.sign ^ edge->label_.sign) & mask & edge->label_.mask) == 0) {
                        label = label | edge->label_;
                        add_edge(source, final, label);
                    }
                }
 
                mask = ~(((label_size_t) 1 << spec->num_inputs_) - 1);
                label = edge->label_.shift(spec->num_outputs_, mask);
                label_size_t rec_bit = ((label_size_t)1) << (num_inputs_ + num_outputs_ - 1);
                const label_t recovery_pos(rec_bit, rec_bit);
                const label_t recovery_neg(rec_bit, 0);

                // add edges leading out of recovery zone
                add_edge(find_node_by_id(source->id_ + spec->nodes_.size()), target, label.merge(recovery_neg));

                // add edge inside recovery zone
                add_edge(find_node_by_id(source->id_ + spec->nodes_.size()), find_node_by_id(target->id_ + spec->nodes_.size()), label.merge(recovery_pos));
            }
        }
    }
    
    cleanup_nodes();
    minimize();
}