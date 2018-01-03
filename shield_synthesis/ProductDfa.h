#include "Dfa.h"
#include <typeinfo>
#include <iostream>
#include <exception>
#include <vector>
#include <list>

class ProductDfa : public Dfa
{
public:
    class ProductDfaNode : public Dfa::Node
    {
    public:
        std::vector<Node*> subnodes_;
        
        ProductDfaNode() :
        Dfa::Node(-1)
        {
        }
        
        ProductDfaNode(ProductDfaNode* node) :
        Dfa::Node(-1)
        {
            initial_ = node->initial_;
            final_ = node->final_;
            
            for (auto subnode : node->subnodes_) {
                subnodes_.push_back(subnode);
            }
            
            for (auto edge : node->edges_) {
                add_edge(new Edge(edge->target_, edge->label_));
            }
            
            final_ = node->final_;
            initial_ = node->initial_;
        }
        
        ~ProductDfaNode() {
            subnodes_.clear();
        }
        
        std::string to_string();
        
        bool operator==(Node& other) {
            try {
                ProductDfaNode* prod_node = dynamic_cast<ProductDfaNode*>(&other);
                if (subnodes_.size() != prod_node->subnodes_.size())
                    return false;
                for (int i = 0; i < subnodes_.size(); i++) {
                    if (subnodes_[i] != prod_node->subnodes_[i])
                        return false;
                }
                return true;
            } catch (...) {
                return false;
            }
        }
    };

    ProductDfa(std::vector<Dfa*> args);
    std::vector<Dfa*> create_combinable_dfas(std::vector<Dfa*> originals);
};