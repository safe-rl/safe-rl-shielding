#include "Dfa.h"
#include <typeinfo>
#include <iostream>
#include <exception>

class ShieldMonitor : public Dfa
{
public:
    class ShieldMonitorNode : public Dfa::Node
    {
    public:
        int design_error_;
        
        ShieldMonitorNode(int id, int design_error = 0) :
        Node(id),
        design_error_(design_error)
        {
        }
    };

    ShieldMonitor(Dfa* spec);
};