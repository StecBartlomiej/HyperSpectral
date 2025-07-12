#ifndef HYPERSPECTRAL_ENTITYCOMPONENTSYSTEM_HPP
#define HYPERSPECTRAL_ENTITYCOMPONENTSYSTEM_HPP

#include "../Logger.hpp"

#include <stack>
#include <set>
#include <bitset>
#include <unordered_map>
#include <array>
#include <typeindex>


using Entity = uint16_t;
using ComponentType = uint8_t;

// List of attributes
using Attributes = std::bitset<32>;

inline constexpr std::size_t max_entities = 10000;
inline constexpr std::size_t max_components = 32;


class EntityManager
{
public:
    EntityManager();

    [[nodiscard]] Entity CreateEntity();

    void DeleteEntity(Entity entity);

    void AssignAttributes(Entity entity, Attributes attributes);

    [[nodiscard]] Attributes GetAttributes(Entity entity);

private:
    std::stack<Entity> available_entity_;
    std::unordered_map<Entity, Attributes> entity_to_attributes_;
    std::size_t living_entity_count_ = 0;
};


class IComponentArray
{
public:
    virtual ~IComponentArray() = default;
    virtual void DeleteEntity(Entity entity) = 0;
};

template <typename T>
class ComponentArray final: public IComponentArray
{
public:
    void Add(Entity entity, T component)
    {
        assert(entity_to_idx_.find(entity) == entity_to_idx_.end() ||
               [entity](){ LOG_ERROR("ComponentArray::Add: Adding already exising entity={}", entity); return false;}());

        component_array_[size_] = component;
        entity_to_idx_[entity] = size_;
        idx_to_entity_[size_] = entity;
        ++size_;
    }

    T& Get(Entity entity)
    {
        auto iter = entity_to_idx_.find(entity);
        assert(iter != entity_to_idx_.end() ||
               [entity](){ LOG_ERROR("ComponentArray::Get: Accessing non-exising entity={}", entity); return false;}());
        return component_array_[iter->second];
    }

    void DeleteData(Entity entity)
    {
        auto iter = entity_to_idx_.find(entity);
        DeleteData(iter);
    }

    void DeleteData(const std::unordered_map<Entity, std::size_t>::iterator &iter)
    {
        assert(iter != entity_to_idx_.end() ||
               [](){ LOG_ERROR("ComponentArray::DeleteData: deleting non-exisintg iterator"); return false;}());

        auto entity = iter->first;
        auto deleted_idx = iter->second;

        --size_;
        component_array_[deleted_idx] = component_array_[size_];

        Entity moved_entity = idx_to_entity_[size_];
        entity_to_idx_[moved_entity] =  deleted_idx;
        idx_to_entity_[deleted_idx] = moved_entity;

        entity_to_idx_.erase(entity);
        idx_to_entity_.erase(size_);
    }

    void DeleteEntity(Entity entity) override
    {
        auto iter = entity_to_idx_.find(entity);
        if (iter != entity_to_idx_.end())
        {
            DeleteData(iter);
        }
    }

private:
    std::array<T, max_components> component_array_;

    std::unordered_map<Entity, std::size_t> entity_to_idx_;
    std::unordered_map<std::size_t, Entity> idx_to_entity_;

    std::size_t size_ = 0;
};


class ComponentManger
{
public:
    template<typename T>
    void RegisterComponent()
    {
        auto type_idx = std::type_index(typeid(T));

        assert(type_to_component_.find(type_idx) == type_to_component_.end() ||
               [](){ LOG_ERROR("ComponentManager::RegisterComponent: tried to register already existing component"); return false;}());

        type_to_component_.insert({type_idx, next_component_type_});
        ++next_component_type_;

        component_arrays_.insert({type_idx, std::make_unique<ComponentArray<T>>()});
    }

    template<typename T>
    [[nodiscard]] ComponentType GetComponentType() const
    {
        auto type_idx = std::type_index(typeid(T));
        auto iter = type_to_component_.find(type_idx);

        assert(iter != type_to_component_.end() ||
               [](){ LOG_ERROR("ComponentManager::GetComponentType: tried to get non-registered component type"); return false;}());

        return iter->second;
    }

    template<typename T>
    void AddComponent(Entity entity, T component)
    {
        GetArray<T>()->Add(entity, component);
    }

    template<typename T>
    void DeleteComponent(Entity entity)
    {
        GetArray<T>()->DeleteData(entity);
    }

    template<typename T>
    [[nodiscard]] T& GetComponent(Entity entity)
    {
        return GetArray<T>()->Get(entity);
    }

    void DeleteEntity(Entity entity);

private:
    std::unordered_map<std::type_index, ComponentType> type_to_component_;
    ComponentType next_component_type_ = 0;

    std::unordered_map<std::type_index, std::unique_ptr<IComponentArray>> component_arrays_;

    template<typename T>
    ComponentArray<T>* GetArray()
    {
        auto type_idx = std::type_index(typeid(T));
        auto iter = component_arrays_.find(type_idx);

        assert(iter != component_arrays_.end() ||
               [](){ LOG_ERROR("ComponentManager::GetArray: tired to access non-existing array for component"); return false;}());

        return static_cast<ComponentArray<T>*>(iter->second.get());
    }
};

class System
{
public:
    std::set<Entity> entities_;
};


class SystemManager
{
public:
    template<typename T>
    [[nodiscard]] T* RegisterSystem()
    {
        auto type_idx = std::type_index(typeid(T));

        assert(systems_.find(type_idx) == systems_.end() ||
               [](){ LOG_ERROR("SystemManager::RegisterSystem: tried to register already existing system"); return false;}());

        systems_[type_idx] = std::make_unique<T>();
        return static_cast<T*>(systems_[type_idx].get());
    }

    template<typename T>
    void SetAttributes(Attributes attributes)
    {
        auto type_idx = std::type_index(typeid(T));

        assert(systems_.find(type_idx) != systems_.end() ||
               [](){ LOG_ERROR("SystemManager::SetAttribute: tried to set attributes of non-existing system"); return false;}());

        attributes_[type_idx] = attributes;
    }

    void DeleteEntity(Entity entity);

    void EntityAttributesChanged(Entity entity, Attributes attributes);

private:
    // maps System to required Components
    std::unordered_map<std::type_index, Attributes> attributes_;

    std::unordered_map<std::type_index, std::unique_ptr<System>> systems_;

};


class Coordinator
{
public:
    Coordinator();

    [[nodiscard]] Entity CreateEntity();

    void DestroyEntity(Entity entity);

    template<typename T>
    void RegisterComponent()
    {
        component_manager_->RegisterComponent<T>();
    }

    template<typename T>
    void AddComponent(Entity entity, T component)
    {
        component_manager_->AddComponent<T>(entity, component);

        auto attributes = entity_manager_->GetAttributes(entity);
        attributes.set(component_manager_->GetComponentType<T>(), true);

        entity_manager_->AssignAttributes(entity, attributes);
        system_manager_->EntityAttributesChanged(entity, attributes);
    }

    template<typename T>
    void DeleteComponent(Entity entity)
    {
        component_manager_->DeleteComponent<T>(entity);

        auto attributes = entity_manager_->GetAttributes(entity);
        attributes.set(component_manager_->GetComponentType<T>(), false);

        entity_manager_->AssignAttributes(entity, attributes);
        system_manager_->EntityAttributesChanged(entity, attributes);
    }

    template<typename T>
    [[nodiscard]] T& GetComponent(Entity entity)
    {
        return component_manager_->GetComponent<T>(entity);
    }

    template<typename T>
    [[nodiscard]] ComponentType GetComponentType()
    {
        return component_manager_->GetComponentType<T>();
    }

    template<typename T>
    T* RegisterSystem()
    {
        return system_manager_->RegisterSystem<T>();
    }

    template<typename T>
    void SetSystemAttribute(Attributes attributes)
    {
        system_manager_->SetAttributes<T>(attributes);
    }

private:
    std::unique_ptr<EntityManager> entity_manager_;
    std::unique_ptr<ComponentManger> component_manager_;
    std::unique_ptr<SystemManager> system_manager_;
};

#endif //HYPERSPECTRAL_ENTITYCOMPONENTSYSTEM_HPP
