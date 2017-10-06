#ifndef __TOWER_H__
#define __TOWER_H__

#include "Sprite.h"
#include <string>

//forward decl;
class cppSimulatorImp;

class Tower :public Sprite {
public:
    Tower(cppSimulatorImp* _Engine,
        Side _side, std::string type_name, pos_tup init_loc);
    ~Tower();
    virtual void step();
    virtual void draw();

    static void initTowers(cppSimulatorImp* Engine);

private:
    pos_tup init_loc;
    double viz_radius;
    std::string color;
};

#endif//__TOWER_H__