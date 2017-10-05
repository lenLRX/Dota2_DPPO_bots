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
    inline PyObject* get_canvas() const { return canvas; }
    inline const std::list<Sprite*>& get_sprites() { return Sprites; }
    inline double get_deltatick() const { return delta_tick; }
    inline std::priority_queue<Event>& get_queue() { return queue; }
    void loop();
private:
    cppSimulatorObject* self;
    double tick_time;
    double tick_per_second;
    double delta_tick;
    PyObject* canvas;
    std::list<Sprite*> Sprites;
    std::list<Sprite*> allSprites;
    std::priority_queue<Event> queue;
};

#endif//__SIMULATORIMP_H__