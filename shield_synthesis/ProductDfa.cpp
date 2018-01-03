#include "ProductDfa.h"
#include <iostream>

ProductDfa::ProductDfa(std::vector<Dfa*> args) :
Dfa(0, 0, 0)
{
    std::vector<Dfa*> dfas = create_combinable_dfas(args);
    
    std::list<ProductDfaNode*> working_set;
    std::set<Node*> done;
    std::set<Node*> to_delete;

    for (Dfa* dfa : dfas) {       
        std::vector<Node*> init_nodes = initial_nodes();
        if (init_nodes.empty()) {
            for (auto node : dfa->initial_nodes()) {
                ProductDfaNode* p_node = new ProductDfaNode();
                p_node->subnodes_.push_back(node);
                p_node->initial_ = true;
                add_node(p_node);
            }
        } else {
            for (auto main_node : init_nodes) {
                for (auto node : dfa->initial_nodes()) {
                    ProductDfaNode* p_node = new ProductDfaNode((ProductDfaNode*)main_node);
                    p_node->subnodes_.push_back(node);
                    p_node->initial_ = true;
                    add_node(p_node);
                }
            }
            
            for (auto node : init_nodes) {
                node->initial_ = false;
                to_delete.insert(node);
            }
        }
        
        for (auto node : initial_nodes()) {
            std::cout << "Intial node: " << node->to_string() << std::endl;
            working_set.push_back((ProductDfaNode*)node);
        }
        
        while (!working_set.empty()) {
            ProductDfaNode* node = working_set.front();
            working_set.pop_front();
            if (done.find(node) != done.end()) {
                continue;
            }
            
            done.insert(node);
                        
            Node* subnode = node->subnodes_.back(); // newly added subnode .. edges have to be added
            
            std::set<ProductDfaNode*> to_process;
            if (node->edges_.empty()) {
                for (auto edge : subnode->edges_) {
                    ProductDfaNode* p_node = new ProductDfaNode();
                    p_node->subnodes_.push_back(edge->target_);
                    if (edge->target_->final_) {
                        p_node->final_ = true;
                    }
                    p_node = (ProductDfaNode*) add_node(p_node);
                    add_edge(node, p_node, edge->label_);
                
                    to_process.insert(p_node);
                }
            } else {
                std::set<Edge*> edges = node->edges_;
                for (auto edge : edges) {
                    
                    for (auto subedge : subnode->edges_) {
                        label_t combined = edge->label_.merge(subedge->label_);
                        if (combined.valid) {
                            ProductDfaNode* p_node = new ProductDfaNode((ProductDfaNode*) edge->target_);
                            p_node->subnodes_.push_back(subedge->target_);
                            if (subedge->target_->final_) {
                                p_node->final_ = true;
                            }
                            p_node = (ProductDfaNode*) add_node(p_node);
                            add_edge(node, p_node, combined);
                    
                            to_process.insert(p_node);
                        }
                    }
                
                    to_delete.insert(edge->target_); // will delete all incoming edges
                }
            }
            
            for (auto new_node : to_process) {
                if (done.find(new_node) == done.end()) {                    
                    working_set.push_back(new_node);
                }
            }
            
            
        }        
    
        for (auto old_node : to_delete) {
            remove_node(old_node);
        }
        to_delete.clear();
    }  
    
    
    verify();
    minimize();
}

std::vector<Dfa*> ProductDfa::create_combinable_dfas(std::vector<Dfa*> originals) {
    std::map<int, std::string> joint_input_vars;
    std::map<int, std::string> joint_output_vars;
    for (Dfa* dfa : originals) {
        for (int i = 1; i <= dfa->num_inputs_; i++) {
            int index = -1;
            for (auto pair : joint_input_vars) {
                if (pair.second == dfa->names_[i]) {
                    index = pair.first;
                    break;
                }
            }
            
            if (index == -1) {
                joint_input_vars[joint_input_vars.size() + 1] = dfa->names_[i];
            }
        }
    }
    for (Dfa* dfa : originals) {
        for (int i = dfa->num_inputs_ + 1; i <= dfa->num_inputs_+ dfa->num_outputs_; i++) {
            int index = -1;
            for (auto pair : joint_output_vars) {
                if (pair.second == dfa->names_[i]) {
                    index = pair.first;
                    break;
                }
            }
        
            if (index == -1) {
                joint_output_vars[joint_input_vars.size() + joint_output_vars.size() + 1] = dfa->names_[i];
            }
        } 
    }
    
    for (auto in : joint_input_vars) {
        std::cout << "IN: " << in.second << std::endl;
    }
    
    for (auto in : joint_output_vars) {
        std::cout << "OUT: " << in.second << std::endl;
    }
    
    std::vector<Dfa*> converted;
    for (Dfa* dfa : originals) {
        converted.push_back(dfa->clone_with_joint_variables(joint_input_vars, joint_output_vars));
    }
    
    num_inputs_ = joint_input_vars.size();
    num_outputs_ = joint_output_vars.size();
    names_.insert(joint_input_vars.begin(), joint_input_vars.end());
    names_.insert(joint_output_vars.begin(), joint_output_vars.end());
   
    return converted;
}

std::string ProductDfa::ProductDfaNode::to_string() {
    std::stringstream stream;
    stream << "<" << id_ << "> or <[";
    for (auto node : subnodes_) {
        stream << node->to_string() << ",";
    }
    stream << "]>";
    return stream.str();
}
