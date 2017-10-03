#ifndef __SIMULATORIMP_H__
#define __SIMULATORIMP_H__

#include "simulator.h"
#include "Sprite.h"
#include "Event.h"
#include <queue>
#include <list>

class cppSimulatorImp
{
public:
    cppSimulatorImp() = delete;
    cppSimulatorImp(cppSimulatorObject* obj,PyObject* canvas = nullptr);
    ~cppSimulatorImp();
    double get_time();
    inline void tick_tick() { tick_time += delta_tick; }
    inline void addSprite(Sprite* s) { Sprites.push_back(s); }
    void loop();
private:
    cppSimulatorObject* self;
    double tick_time;
    double tick_per_second;
    double delta_tick;
    PyObject* canvas;
    std::list<Sprite*> Sprites;
    std::priority_queue<Event> queue;
};

#endif//__SIMULATORIMP_H__