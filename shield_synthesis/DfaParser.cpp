#include "DfaParser.h"
#include "Dfa.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

Dfa* DfaParser::parse_dfa_from_file(std::string file_name)
{
    std::string in = " ";
    std::string test = trim(in);
    
    std::ifstream file(file_name);
    if (!file.is_open()) {
        return NULL;
    }
    
    std::stringstream cleaned;
    std::string line;
    while(std::getline(file, line))
    {
        line = strip_comments(trim(line));
        if (!line.empty()) cleaned << line << std::endl;
        
    }
    
    std::getline(cleaned, line);
    std::vector<std::string> parts = split(line);
    // std::cout << line << std::endl;
    if (parts.size() != 7 || parts[0] != "dfa") {
        std::cout << "Parsing Error: Input file has to start with 'dfa' followed by 6 integers\n";
        file.close();
        return NULL;
    }  
    
    int num_inputs = std::stoi(parts[2]), num_outputs = std::stoi(parts[3]);
    Dfa* dfa = new Dfa(std::stoi(parts[1]), num_inputs, num_outputs);
    int num_edges = std::stoi(parts[6]);
    
    std::getline(cleaned, line);
    parts = split(line);
    for (auto initial_state : parts) {
        dfa->find_node_by_id(std::stoi(initial_state))->initial_ = true;
    }
    
    std::getline(cleaned, line);
    parts = split(line);
    for (auto final_state : parts) {
        dfa->find_node_by_id(std::stoi(final_state))->final_ = true;
    }  
    
    for (int i = 0; i < num_edges; i++) {
        if (!std::getline(cleaned, line)) {
            std::cout << "Parsing Error: Not enough edges!\n";
            delete dfa;
            return NULL;
        }        
        
        parts = split(line);
        if (parts.size() < 2) {
            std::cout << "Parsing Error: Edges must at least contain two states\n";
            delete dfa;
            return NULL;
        }
        
        int source = std::stoi(parts[0]);
        int target = std::stoi(parts[1]);
        parts.erase(parts.begin());
        parts.erase(parts.begin());
        
        label_t label;
        for (auto literal : parts) {
            int lit = std::stoi(literal);
            label.mask |= ((label_size_t)1U) << (std::abs(lit) - 1);
            if (lit > 0) label.sign |= ((label_size_t)1U) << (std::abs(lit) - 1);
        }
        
        dfa->add_edge(source, target, label);
    }
    
    for (int i = 0; i < num_inputs + num_outputs; i++) {
        std::getline(cleaned, line);
        parts = split(line);
        dfa->set_variable_name(std::stoi(parts[0]), parts[1]);
    }
    
    dfa->minimize();
    return dfa;
}
