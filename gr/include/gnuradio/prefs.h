#pragma once


#include <yaml-cpp/yaml.h>
#include <cstdio>  //P_tmpdir (maybe)
#include <cstdlib> //getenv
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;


namespace gr {

class prefs
{
public:
    static prefs* get_instance()
    {
        static prefs p;
        return &p;
    }

    static YAML::Node get_section(const std::string& name)
    {
        return get_instance()->_config[name];
    }

    static const std::string get_string(const std::string& section,
                                        const std::string& option,
                                        const std::string& default_val)
    {
        try {
            auto s = get_section(section);
            return s[option].as<std::string>();
        } catch (const std::exception& e) {
            return default_val;
        }
    }

    static const char* appdata_path()
    {
        const char* path;

        // first case, try HOME environment variable (unix)
        path = getenv("HOME");
        if (path)
            return path;

        // second case, try APPDATA environment variable (windows)
        path = getenv("APPDATA");
        if (path)
            return path;

        // fall-through case, nothing worked
        return tmp_path();
    }

private:
    YAML::Node _config;

    prefs()
    {
        std::vector<std::string> fnames;

        // find the preferences yaml file
        // Find if there is a ~/.gnuradio/config.conf file
        fs::path userconf = fs::path(userconf_path()) / "config.yml";
        if (fs::exists(userconf)) {
            // fnames.push_back(userconf.string());
            _config = YAML::LoadFile(userconf.string());
        }
    }

    static const char* tmp_path()
    {
        const char* path;

        // first case, try TMP environment variable
        path = getenv("TMP");
        if (path)
            return path;

// second case, try P_tmpdir when its defined
#ifdef P_tmpdir
        if (P_tmpdir)
            return P_tmpdir;
#endif /*P_tmpdir*/

        // fall-through case, nothing worked
        return "/tmp";
    }


    std::string __userconf_path()
    {
        const char* path;

        // First determine if there is an environment variable specifying the prefs path
        path = getenv("GR_PREFS_PATH");
        fs::path p;
        if (path) {
            p = path;
        }
        else {
            p = appdata_path();
            p = p / ".gnuradio";
        }

        return p.string();
    }

    const char* userconf_path()
    {
        static std::string p(__userconf_path());
        return p.c_str();
    }
};

} // namespace gr