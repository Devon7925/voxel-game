# Voxel Game design document

## Principles
The primary game loop is design sets of gun decks using cards and then use those decks to fight. Because deckbuilding is a large part of the game the gun decks must therefore allow for a large amount of player expression in order to be interesting. With that in mind cards are relitively simple. However in order for that player expression to have a meaningful impact it must also allow the player to do a large amount of things which is why there is a destructable/constructable world that the player can interact with using the gun decks.

## Cards

### BaseCard
- Projectile(ProjectileModifier[])
- Multicast(BaseCard[])
- Effect(Effect)
- Material(Material)

### ProjectileModifier
- Speed
- Size
- Lifetime
- Health
- Over time(ProjectileModifier)
- Add gravity
- No friendly fire
- Only friendly fire
- Piercing
- Trail(BaseCard)
- On hit(BaseCard)
- No collision

### Effect
- Damage
- Knockback
- Damage recieved
- Damage dealt
- Movement speed
- Attack speed

## Materials

- Rock
- Dirt
- Grass
- Ice
- Glass
- Indestructible
- Air


