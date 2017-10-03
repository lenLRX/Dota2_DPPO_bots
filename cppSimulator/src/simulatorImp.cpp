#include "simulatorImp.h"
#include "Config.h"

cppSimulatorImp::cppSimulatorImp(cppSimulatorObject* obj, PyObject* canvas)
    :self(obj), tick_time(0.0), tick_per_second(Config::tick_per_second),
    canvas(canvas)
{

}

cppSimulatorImp::~cppSimulatorImp()
{
    for (auto p : Sprites) {
        delete p;
    }
}

double cppSimulatorImp::get_time()
{
    return tick_time;
}

void cppSimulatorImp::loop()
{
    while (!queue.empty() &&
        queue.top().get_time() < tick_time)
    {
        auto event = queue.top();
        queue.pop();
        event.activate();
    }

}
