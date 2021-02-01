#include "ProductDfa.h"
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <vector>

ProductDfa::ProductDfa(std::vector<Dfa*> args) :
// TODO fdh: possible optimizations:
//  * skip all nodes that are not connected / not connected to initial nodes
//  * reduce number of for-loops by setting initial_, final_ and edges in single for-loop
Dfa(0, 0, 0)
{
    // create combinable dfas by renaming variables etc. 
    std::vector<Dfa*> dfas = create_combinable_dfas(args);

    // Extract nodes from all dfas.
    // These form the underlying structure of the final dfa
    std::vector<std::vector<Node*>> subnodes;
    for (Dfa* dfa : dfas) {
		std::vector<Node*> v;
	    for( auto node : dfa->nodes_){
			v.push_back(node);
		}
        subnodes.push_back(v);
    }
    // Create cartesian product of subnodes of different dfas. Each combination of subnodes from
    // different dfas represents a node in the product dfa
    std::vector<std::vector<Node*>> node_combinations = create_cartesian(subnodes);
    // create pdfa node w/ subnodes for each element in cartesian product
    for (auto node_v : node_combinations){
        // create a new ProductDfaNode and fill it
        ProductDfaNode* pdfa_node = new ProductDfaNode();
        std::set<Node*> subnode_set;
        for(auto node : node_v){
            pdfa_node->subnodes_.push_back(node);
            subnode_set.insert(node);
        }
        // add the pdfa node to the pdfa
        pdfa_node = (ProductDfaNode*) add_node(pdfa_node);
        // store the subnodes that form this pdfa node in a map
        subnode_node_map[subnode_set] = pdfa_node;
    }

    // set pdfa node initial flag *only* if all subnodes are initial
    for (Node* node : nodes_){
        ProductDfaNode* pdfa_node = (ProductDfaNode*) node;
        bool initial_ = true;
        for( auto n : pdfa_node->subnodes_){
            if(!n->initial_){
                initial_ = false;
                break;
            }
        }
        pdfa_node->initial_ = initial_;
    }

    // Set pdfa node final flag if *a* subnode is final. These are accepting states/nodes.
    for (Node* node : nodes_){
        ProductDfaNode* pdfa_node = (ProductDfaNode*) node;
        bool final_ = false;
        for( auto n : pdfa_node->subnodes_){
            if(n->final_){
                final_ = true;
                break;
            }
        }
        pdfa_node->final_ = final_;
    }

    // Combine nodes edges
    for (Node* node : nodes_){
        ProductDfaNode* pdfa_node = (ProductDfaNode*) node;
        // create a set of edges for all of this pdfa nodes' underlying nodes edges
        std::vector<std::vector<Dfa::Edge*>> combo_edges;
        // loop all subnodes/underlying nodes
        for (auto subnode : pdfa_node->subnodes_){
            std::vector<Edge*> edges;
            // extract all edges
            for( auto edge : subnode->edges_){
                edges.push_back(edge);
            }
            combo_edges.push_back(edges);
        }
        // create a cartesian product of outgoing edges for all subnodes of this pdfa node.
        // Following these edges/transitions 'jointly' should make us end up in the relating pdfa
        // state representing all underlying target nodes.
        std::vector<std::vector<Edge*>> cart_edge = create_cartesian(combo_edges);
        // loop over all 'joint' edges/transitions
        for (auto combo_edge : cart_edge){
            // collect all underlying target nodes in a set
            std::set<Node*> subnode_targets;
            label_t label;
            // combine the labels for each transition into a single label
            for(auto edge : combo_edge){
                label = label.merge(edge->label_);
                if(!label.valid){
                    // TODO: how to deal with invalid label? This would mean that the input dfas
                    // cannot be combined?
                    std::cout << "Invalid label combination!" << std::endl;
                    exit(EXIT_FAILURE);
                }
                subnode_targets.insert(edge->target_);
            }
            // find target pdfa node by following underlying node set in 'reverse' subnode->node map
            ProductDfaNode* target_node = subnode_node_map[subnode_targets];
            // insert link from pdfa node to pdfa node w/ combined label
            add_edge(pdfa_node, target_node, label);
        }
    }

    // create all combinations of subnodes, pegging on some (the last) dfa 
    // two inputs
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

std::vector<std::vector<Dfa::Edge*>> ProductDfa::create_cartesian( std::vector<std::vector<Edge*>>& v){
    std::vector<std::vector<Edge*>> s = {{}};
    for( const auto& u : v){
        std::vector<std::vector<Edge*>> r;
        for( const auto& x: s){
            for( const auto y: u){
                r.push_back(x);
                r.back().push_back(y);
            }
        }
        s = move(r);
    }
    return s;
}


std::vector<std::vector<Dfa::Node*>> ProductDfa::create_cartesian( std::vector<std::vector<Node*>>& v){
    std::vector<std::vector<Node*>> s = {{}};
    for( const auto& u : v){
        std::vector<std::vector<Node*>> r;
        for( const auto& x: s){
            for( const auto y: u){
                r.push_back(x);
                r.back().push_back(y);
            }
        }
        s = move(r);
    }
    return s;
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
