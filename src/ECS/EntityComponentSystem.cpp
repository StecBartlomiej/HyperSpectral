#include "EntityComponentSystem.hpp"

#include <cassert>


EntityManager::EntityManager()
{
    for (std::size_t i = 0; i < max_entities; ++i)
    {
        available_entity_.push(max_entities - i - 1);
    }
}

Entity EntityManager::CreateEntity()
{
    assert(!available_entity_.empty() ||
            [](){ LOG_ERROR("EntityManager::CreateEntity: cant create Entity, all id's are assigned."); return false;}());

    auto id = available_entity_.top();
    available_entity_.pop();
    entity_to_attributes_[id] = 0;

    ++living_entity_count_;

    return id;
}

void EntityManager::DeleteEntity(Entity entity)
{
    auto idx = entity_to_attributes_.find(entity);
    assert(idx == entity_to_attributes_.end() || [entity](){
        LOG_ERROR("EntityManager::DeleteEntity: deleting non-existing entity with id={} ", entity); return false; }());

    entity_to_attributes_.erase(idx);

    --living_entity_count_;
    available_entity_.push(entity);
}

void EntityManager::AssignAttributes(Entity entity, Attributes attributes)
{
    auto idx = entity_to_attributes_.find(entity);
    assert(idx != entity_to_attributes_.end() || [entity](){
        LOG_ERROR("EntityManager::AssignAttributes: assign attributes to non-existing entity with id={} ", entity);
        return false; }());

    idx->second = attributes;
}

Attributes EntityManager::GetAttributes(Entity entity)
{
    return entity_to_attributes_[entity];
}

void ComponentManger::DeleteEntity(Entity entity)
{
    for (const auto &iter: component_arrays_)
    {
        iter.second->DeleteEntity(entity);
    }
}

void SystemManager::DeleteEntity(Entity entity)
{
    for (auto &iter : systems_)
    {
        iter.second->entities_.erase(entity);
    }
}

void SystemManager::EntityAttributesChanged(Entity entity, Attributes attributes)
{
    for (auto &iter : systems_)
    {
        auto type_idx = iter.first;
        auto &system = iter.second;
        auto system_attributes = attributes_[type_idx];

        if ((system_attributes & attributes) == system_attributes)
        {
            system->entities_.insert(entity);
        }
        else
        {
            system->entities_.erase(entity);
        }
    }
}

Coordinator::Coordinator(): entity_manager_{std::make_unique<EntityManager>()},
                            component_manager_{std::make_unique<ComponentManger>()},
                            system_manager_{std::make_unique<SystemManager>()}
{
}

Entity Coordinator::CreateEntity()
{
    return entity_manager_->CreateEntity();
}

void Coordinator::DestroyEntity(Entity entity)
{
    entity_manager_->DeleteEntity(entity);
    component_manager_->DeleteEntity(entity);
    system_manager_->DeleteEntity(entity);
}
