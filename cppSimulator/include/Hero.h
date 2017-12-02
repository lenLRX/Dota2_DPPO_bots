#ifndef __HERO_H__
#define __HERO_H__

#include "Sprite.h"

#include <string>

//forward decl;
class cppSimulatorImp;

enum decisonType {
    noop = 0,
    move = 1,
    attack = 2
};

using target_list_t = std::vector<Sprite*>;

class Hero:public Sprite {
public:
    Hero(cppSimulatorImp* _Engine,
        Side _side, std::string type_name);
    ~Hero();
    virtual void step();
    virtual void draw();
    void set_order(PyObject* order);
    PyObject* get_state_tup();
    PyObject* predefined_step();
private:
    pos_tup init_loc;
    pos_tup move_order;
    int decision;
    Sprite* target;
    target_list_t target_list;
    std::vector<target_list_t> histroy_target_lists;
    double viz_radius;
    double last_gold;
    double last_exp;
    double last_HP;
    std::string color;
};

#endif//__HERO_H__