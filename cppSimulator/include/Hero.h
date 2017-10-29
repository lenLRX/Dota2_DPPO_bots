#ifndef __HERO_H__
#define __HERO_H__

#include "Sprite.h"

#include <string>

//forward decl;
class cppSimulatorImp;

class Hero:public Sprite {
public:
    Hero(cppSimulatorImp* _Engine,
        Side _side, std::string type_name);
    ~Hero();
    virtual void step();
    virtual void draw();
    void set_move_order(pos_tup order);
    PyObject* get_state_tup();
    PyObject* predefined_step();
private:
    pos_tup init_loc;
    pos_tup move_order;
    double viz_radius;
    double last_exp;
    double last_HP;
    std::string color;
};

#endif//__HERO_H__