#ifndef DFA_H
#define DFA_H

#include <vector>
#include <map>
#include <string>
#include <set>
#include <sstream>
#include <iostream>
#include <initializer_list>
#include <bitset>

typedef uint64_t label_size_t;
typedef struct label {
    label_size_t mask;
    label_size_t sign;
    bool valid;
    
    label() {
        mask = 0U;
        sign = 0U;
        valid = true;
    }
    
    label(label_size_t m, label_size_t s) {
        mask = m;
        sign = s;
        valid = true;
    }
    
    bool operator ==(const label& other) {
        return mask == other.mask && ((sign ^ other.sign) & mask) == 0;
    }
    
    label operator &(const label& other) {
        label res;
        res.mask = mask & other.mask;
        res.sign = sign & other.sign;
        return res;
    }
    
    label operator &(const label_size_t& other_mask) {
        label res;
        res.mask = mask & other_mask;
        res.sign = sign & other_mask;
        return res;
    }
    
    label operator |(const label& other) {
        label res;
        res.mask = mask | other.mask;
        res.sign = sign | other.sign;
        return res;
    }
    
    label operator <<(const int bits) {
        label res;
        res.mask = mask << bits;
        res.sign = sign << bits;
        res.valid = valid;
        return res;
    }
    
    label operator >>(const int bits) {
        label res;
        res.mask = mask >> bits;
        res.sign = sign >> bits;
        res.valid = valid;
        return res;
    }
    
    label shift(int amount, label_size_t shift_mask = -1U) { 
        return ((*this & shift_mask) << amount) | (*this & ~shift_mask);
    }
    
    label merge(const label& other) {
        label combined;
        combined.mask = mask | other.mask;
        combined.sign = sign | other.sign;
        combined.valid = ((sign ^ other.sign) & (mask & other.mask)) == 0;
        return combined;
    }
    
    void remove(const label& other) {
        mask &= ~other.mask; 
        sign &= ~other.mask;
    }
    
    bool check_compatibility(const label& other, label_size_t input_mask = -1U) {
        return ((sign ^ other.sign) & (mask & other.mask & input_mask))  == 0;
    }
    
} label_t;

class Dfa
{
public:
    
    class Edge;
    class Node {
    public:
        std::set<Edge*> edges_;
        int id_;
        bool initial_, final_;
       
        Node(int id) :
        id_(id),
        initial_(false),
        final_(false)
        {
        }
        
        Node(Node* other) :
        id_(other->id_),
        initial_(other->initial_),
        final_(other->final_)
        {
        }
        
        virtual ~Node() {
            for (Edge* edge : edges_) {
                if (edge) delete edge;
            }
            edges_.clear();
        };
        
        virtual bool operator==(Node& other) {
            return id_ == other.id_;
        }
        
        virtual std::string to_string();
        
        Edge* add_edge(Edge* edge);
        void remove_edge(Edge* edge);
    };
    
    class Edge {
    public:
        Node* target_;
        label_t label_;
        
        Edge(Node* target, label_t label) {
            target_ = target;
            label_ = label;
        }
        
        bool operator==(Edge& other) {
            return (*target_ == *other.target_) && (label_ == other.label_);
        }
        
        std::string to_string(std::map<int, std::string>& variable_names) {
            std::stringstream stream;
            
            for (int mask = 1U, idx = 0; mask <= label_.mask; mask <<= 1, idx++ ) {
                if ((label_.mask & mask) == mask) {
                    stream << (((label_.sign & mask) == mask) ? "" : "!") << variable_names[idx + 1] << " | ";
                }
            }
            return stream.str();
        }
        
        ~Edge() {
            
        }
    };
    
public:
    int num_states_;
    int num_inputs_;
    int num_outputs_;  
    std::set<Node*> nodes_;
    std::map<int, std::string> names_;
public:
    Dfa(int num_states, int num_inputs, int num_outputs);
    Dfa(std::initializer_list<Dfa*> args);
    Node* add_node(Node* node);
    void remove_node(Node* node);
    void cleanup_nodes();
    Edge* add_edge(int source, int target, label_t label);
    void add_edges(int source, int target, std::vector<label_t> labels);
    Edge* add_edge(Node* source, Node* target, label_t label);
    void add_edges(Node* source, Node* target, std::vector<label_t> labels);
    void set_variable_name(int index, std::string name);
    bool check_label_compatibility(label_t& l1, label_t& l2);
    Node* find_node_by_id(int id);
    std::vector<Node*> final_nodes();
    std::vector<Node*> initial_nodes();
    std::string to_string();
    void minimize();
    void exchange_target_node(Node* old_node, Node* new_node);
    void verify();
    Dfa* clone_with_joint_variables(std::map<int, std::string> inputs, std::map<int, std::string> outputs);
    
    inline int count_bits(label_size_t n) {        
        uint_fast32_t count = 0;
        for (; n; count++) {
            n &= n - 1;
        }
        return count;
    }
};

#endif