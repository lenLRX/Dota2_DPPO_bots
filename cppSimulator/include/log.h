#ifndef __LOG_H__
#define __LOG_H__
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>

using namespace std;

#define LOG Logger::getInstance() <<  Logger::getTime() + string(" ") << __FILE__ << " line " << __LINE__ << ":"
class Logger
{
  public:
    static Logger &getInstance();
    void redirectStream(string filename);
    void flush();
    template <typename T>
    Logger &operator<<(const T &obj)
    {
	stream << obj;
	return *this;
    }

    static string getTime()
    {
        time_t rawtime;
        struct tm * timeinfo;

        time ( &rawtime );
        timeinfo = localtime ( &rawtime );
        return asctime (timeinfo);
    }

    Logger &operator<<(ostream &(*_Pfn)(ostream &));

    ~Logger();

  private:
    fstream _fstream;
    ostream stream;
    Logger();
};

#endif