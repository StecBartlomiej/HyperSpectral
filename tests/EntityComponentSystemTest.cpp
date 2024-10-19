#include <catch2/catch_test_macros.hpp>

#include "EntityComponentSystem.hpp"


Coordinator coordinator{};

struct Position
{
    int x;
    int y;
};

struct RigidBody
{
    int speed_x;
    int speed_y;
};

class PhysicSystem : public System
{
public:
    void Update(int dt)
    {
        for (auto &entity : entities_)
        {
            auto &position = coordinator.GetComponent<Position>(entity);
            auto &rigid_body = coordinator.GetComponent<RigidBody>(entity);

            position.x += rigid_body.speed_x * dt;
            position.y += rigid_body.speed_y * dt;
        }
    }
};

TEST_CASE("ECS basic usage", "[ECS]")
{
    constexpr int x_pos = 10;
    constexpr int y_pos = -1;
    constexpr int x_speed = 1;
    constexpr int y_speed = 2;
    constexpr int dt = 1;

    coordinator.RegisterComponent<Position>();
    coordinator.RegisterComponent<RigidBody>();

    auto *physic_system = coordinator.RegisterSystem<PhysicSystem>();

    Attributes attributes;
    attributes.set(coordinator.GetComponentType<Position>());
    attributes.set(coordinator.GetComponentType<RigidBody>());

    coordinator.SetSystemAttribute<PhysicSystem>(attributes);

    auto entity = coordinator.CreateEntity();

    coordinator.AddComponent(entity, Position{x_pos, y_pos});
    coordinator.AddComponent(entity, RigidBody{x_speed, y_speed});

    physic_system->Update(dt);

    auto position = coordinator.GetComponent<Position>(entity);

    REQUIRE(position.x == x_pos + x_speed * dt);
    REQUIRE(position.y == y_pos + y_speed * dt);
}
