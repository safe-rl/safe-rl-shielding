#include "Dfa.h"
#include <sstream>
#include <iostream>
#include <bitset>
#include <typeinfo>

Dfa::Dfa(int num_states, int num_inputs, int num_outputs) :
num_states_(num_states),
num_inputs_(num_inputs),
num_outputs_(num_outputs)
{
    for (int i = 1; i <= num_states; i++) {
        nodes_.insert(new Node(i));
    }
    
    for (int i = 1; i <= num_inputs; i++) {
        names_[i] = "i" + std::to_string(i); 
    }
    
    for (int i = num_inputs + 1; i <= num_inputs + num_outputs; i++) {
        names_[i] = "o" + std::to_string(i - num_inputs); 
    }
}

Dfa::Edge* Dfa::add_edge(int source, int target, label_t label)
{
    Node* source_node = find_node_by_id(source);
    Node* target_node = find_node_by_id(target);
    return add_edge(source_node, target_node, label);
}

void Dfa::add_edges(int source, int target, std::vector<label_t> labels)
{
    Node* source_node = find_node_by_id(source);
    Node* target_node = find_node_by_id(target);
    add_edges(source_node, target_node, labels);
}

void Dfa::set_variable_name(int index, std::string name) 
{
    names_[index] = name;
}


std::string Dfa::to_string()
{
    std::stringstream stream;
    
    stream << "States: \n";
    for (auto node : nodes_) {
        stream << node->to_string() << (node->initial_ ? " (initial)" : "") << (node->final_ ? " (final)" : "") << std::endl;
    }
    
    stream << "\nTransitions: \n";
    for (auto node : nodes_) {
        for (auto edge : node->edges_) {
            if (edge->target_ != NULL) {
                stream << node->to_string() << " -> " << edge->target_->to_string() << ": " << edge->to_string(names_);
                stream << std::endl;
            }
        }
    }
    
    return stream.str();
}

Dfa::Edge* Dfa::add_edge(Dfa::Node* source, Dfa::Node* target, label_t label)
{
    if (source != NULL && target != NULL)
        return source->add_edge(new Edge(target, label));
    return NULL;
}
 
void Dfa::add_edges(Dfa::Node* source, Dfa::Node* target, std::vector<label_t> labels)
{
    for (label_t label : labels) {
        add_edge(source, target, label);
    }
}

Dfa::Node* Dfa::find_node_by_id(int id)
{    
    for (Node* node : nodes_) {
        if (node->id_ == id) {
            return node;
        }
    }
    return NULL;
}

Dfa::Node* Dfa::add_node(Node* node) 
{
    for (Node* existing : nodes_) {
        if (*node == *existing) {
            if (node != existing) {
                delete node;  
            } 
            return existing;
        }
    }
     
    nodes_.insert(node);
    if (node->id_ == -1) {
        node->id_ = nodes_.size();
    }
    
    return node;
}

void Dfa::remove_node(Dfa::Node* node) 
{
    for (auto other : nodes_) {        
        std::vector<Edge*> to_delete;
        for (auto edge : other->edges_) {
            if (edge->target_ == node) to_delete.push_back(edge);
        }
        
        for (auto edge : to_delete) {
            other->edges_.erase(edge);
            delete edge;
        }
    } 
    
    int deleted_id = -1;
    for (auto other : nodes_) {
        if (*node == *other) {
            deleted_id = other->id_;
            nodes_.erase(other);
            delete other;
            break;
        }
    }
    
    for (auto other : nodes_) {
        if (other->id_ > deleted_id) {
            other->id_--;
        }
    }
}


std::vector<Dfa::Node*> Dfa::final_nodes() 
{
    std::vector<Node*> final;
    for (auto node : nodes_) {
        if (node->final_) final.push_back(node);
    }
    return final;
}

std::vector<Dfa::Node*> Dfa::initial_nodes()
{
    std::vector<Node*> initial;
    for (auto node : nodes_) {
        if (node->initial_) initial.push_back(node);
    }
    return initial;
}

bool Dfa::check_label_compatibility(label_t& l1, label_t& l2)
{
    if (num_inputs_ == 0) return true;
    else return l1.check_compatibility(l2, (((label_size_t)1) << num_inputs_) - 1);
}

void Dfa::minimize()
{   
    for (auto node : nodes_) {
        std::map<Node*, std::vector<Edge*>> sorted;
        for (auto edge : node->edges_) {
            sorted[edge->target_].push_back(edge);
        }
        
        for (auto pair : sorted) {
            bool changed = true;
            while (changed) {
                changed = false;
                int idx_1 = 0;
                while(idx_1 < pair.second.size()) {
                    int idx_2 = idx_1 + 1;
                    Edge* edge_1 = pair.second[idx_1];
                    label_t l1 = edge_1->label_;
                    while(idx_2 < pair.second.size()) {
                        Edge* edge_2 = pair.second[idx_2];
                        label_t l2 = edge_2->label_;
                        if (l1.mask == l2.mask) {
                            if (count_bits((l1.sign ^ l2.sign) & l1.mask) == 1) {                                
                                label_size_t new_mask = l1.mask ^ ((l1.sign ^ l2.sign) & l1.mask);
                                edge_1->label_.mask = new_mask;
                                edge_1->label_.sign &= new_mask;
                                
                                changed = true;
                                pair.second.erase(pair.second.begin() + idx_2);
                                node->remove_edge(edge_2);
                                break;
                            }
                        }
                        idx_2++;
                    }
                    idx_1++;           
                }
            }
        }        
    }
}

void Dfa::cleanup_nodes() 
{
    std::vector<Node*> to_delete;
    for (auto node : nodes_) {
        if (node->edges_.empty()) {
            to_delete.push_back(node);
        }
    }

    for (auto node : to_delete) {
        remove_node(node);
    }
    
    to_delete.clear();
    Node* final = NULL;
    for (auto node : final_nodes()) {
        if (final == NULL) {
            final = node;
        } else {
            exchange_target_node(node, final);
            to_delete.push_back(node);
        }
    }
    
    if (final != NULL) {
        for (auto edge : final->edges_) {
            delete edge;
        }
        final->edges_.clear();
    
        add_edge(final, final, label_t(0,0));
    }    
    
    for (auto node : to_delete) {
        remove_node(node);
    }
}

void Dfa::verify()
{
    for (auto node : nodes_) {
        if (node->final_) {
            continue;
        }
        bool safe = false;
        for (auto edge : node->edges_) {
            if (!edge->target_->final_) {
                // at least one save adjacent state
                safe = true;
                break;
            }
        }
        
        if (!safe) {
            std::cout << "Setting " << node->to_string() << " to final\n";
            node->final_ = true;
        }
    }
    
    cleanup_nodes();
}

void Dfa::exchange_target_node(Node* old_node, Node* new_node)
{
    for (auto node : nodes_) {
        for (auto edge : node->edges_) {
            if (edge->target_ == old_node) {
                edge->target_ = new_node;
            }
        }
    }
}


Dfa::Edge* Dfa::Node::add_edge(Dfa::Edge* edge) 
{
    for (auto existing : edges_) {
        if (*edge == *existing) {
            if (edge != existing) {
                delete edge;  
            } 
            return existing;
        }
    }
    
    edges_.insert(edge);
    return edge;
}

void Dfa::Node::remove_edge(Dfa::Edge* edge)
{
    edges_.erase(edge);
    delete edge;
}


std::string Dfa::Node::to_string() {
    std::stringstream stream;
    stream << "<" << std::to_string(id_) << ">";
    return stream.str();
}

Dfa* Dfa::clone_with_joint_variables(std::map<int, std::string> inputs, std::map<int, std::string> outputs)
{
    Dfa* result = new Dfa(0, inputs.size(), outputs.size());
    
    std::map<int, int> shift_indizes;
    for (int i = 1; i <= num_inputs_; i++) {
        std::string name = names_[i];
        int target_index = -1;
        for (auto pair : inputs) {
            if (pair.second == name) {
                std::cout << name << "has index " << pair.first << std::endl;
                
                target_index = pair.first;
                break;
            }
        }
        
        if (target_index == -1) {
            std::cout << "Aborting: Creating Dfa for joint variable set missing own variable " << name << std::endl;
            return NULL;
        }
            
        shift_indizes[i - 1] = target_index - i;
        std::cout << "Adding shift index for var " << names_[i] << ": " << shift_indizes[i - 1] << std::endl;
    }
    
    
    
    for (int i = num_inputs_ + 1; i <= num_inputs_ + num_outputs_; i++) {
        std::string name = names_[i];
        int target_index = -1;
        for (auto pair : outputs) {
            if (pair.second == name) {
                std::cout << name << "has index " << pair.first << std::endl;
                
                target_index = pair.first;
                break;
            }
        }
        
        if (target_index == -1) {
            std::cout << "Aborting: Creating Dfa for joint variable set missing own variable " << name << std::endl;
            return NULL;
        }
            
        shift_indizes[i - 1] = target_index - i;
        std::cout << "Adding shift index for var " << names_[i] << ": " << shift_indizes[i - 1] << std::endl;
    }
    
    for (auto node : nodes_) {
        result->add_node(new Node(node));
    }
    
    for (auto node : nodes_) {
        for (auto edge : node->edges_) {
            Node* source = result->find_node_by_id(node->id_);
            Node* target = result->find_node_by_id(edge->target_->id_);
            label_t label;
            for (int i = 0; i < num_inputs_ + num_outputs_; i++) {
                label_size_t mask = 1 << i;
                label.mask |= (edge->label_.mask & mask) << shift_indizes[i];
                label.sign |= (edge->label_.sign & mask) << shift_indizes[i];
            }
            result->add_edge(source, target, label);
        }
    }
    
    return result;
}

