#ifndef __SIMULATORIMP_H__
#define __SIMULATORIMP_H__

#include "simulator.h"
#include "Event.h"
#include <queue>
#include <list>
#include <vector>

//forward decl
class Hero;
class Sprite;

class cppSimulatorImp
{
public:
    cppSimulatorImp() = delete;
    cppSimulatorImp(cppSimulatorObject* obj,PyObject* canvas = nullptr);
    ~cppSimulatorImp();
    double get_time();
    inline void tick_tick() { tick_time += delta_tick; }
    inline void addSprite(Sprite* s) { Sprites.push_back(s); allSprites.push_back(s); }
    inline PyObject* get_canvas() const { return canvas; }
    inline const std::list<Sprite*>& get_sprites() { return Sprites; }
    inline double get_deltatick() const { return delta_tick; }
    inline std::priority_queue<Event>& get_queue() { return queue; }
    void loop();
    std::vector<std::pair<Sprite*, double>> get_nearby_enemy(Sprite* s);
    std::vector<std::pair<Sprite*, double>> get_nearby_enemy(Sprite * sprite, std::function<bool(Sprite*)> filter);
    std::vector<std::pair<Sprite*, double>> get_nearby_ally(Sprite* s);
    std::vector<std::pair<Sprite*, double>> get_nearby_ally(Sprite * sprite, std::function<bool(Sprite*)> filter);
    void set_order(PyObject *args, PyObject *kwds);
    PyObject* get_state_tup(std::string side, int idx);
    PyObject* predefined_step(std::string side, int idx);
private:
    cppSimulatorObject* self;
    double tick_time;
    double tick_per_second;
    double delta_tick;
    PyObject* canvas;
    std::vector<Hero*> RadiantHeros;
    std::vector<Hero*> DireHeros;
    std::list<Sprite*> Sprites;
    std::list<Sprite*> allSprites;
    std::priority_queue<Event> queue;
};

#endif//__SIMULATORIMP_H__