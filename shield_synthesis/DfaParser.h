#include <string>
#include <sstream>
#include <vector>

class Dfa;

class DfaParser
{    
    // trim from start
    static inline std::string& ltrim(std::string &s) {
        int pos = s.find_first_not_of(" \t\n\r");
        if (pos != std::string::npos) s = s.substr(pos, s.size() - pos);
        else s = "";
        return s;
    }

    // trim from end
    static inline std::string& rtrim(std::string &s) {
        int pos = s.find_last_not_of(" \t\n\r");
        if (pos != std::string::npos) s = s.substr(0, pos + 1);
        else s = "";
        return s;
    }

    // trim from both ends
    static inline std::string& trim(std::string &s) {
            return ltrim(rtrim(s));
    }
    
    // trim from both ends
    static inline std::string& strip_comments(std::string &s) {
        int pos = s.find_first_of("#");
        if (pos != std::string::npos) s = s.substr(0, pos);
        return rtrim(s);
    }
    
    static inline std::vector<std::string> split(std::string &s) {
        std::stringstream stream(s);
        std::vector<std::string> parts;
        std::string part;
        while(getline(stream, part, ' '))
        {
            if(part.size() > 0) 
            {
                parts.push_back(part);
            }
        }
        return parts;
    }
    
public:
    static Dfa* parse_dfa_from_file(std::string file);
};