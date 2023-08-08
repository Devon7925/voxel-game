# Voxel Game design document

## Principles
The primary game loop is design sets of gun decks using cards and then use those decks to fight. Because deckbuilding is a large part of the game the gun decks must therefore allow for a large amount of player expression in order to be interesting. With that in mind cards are relitively simple. However in order for that player expression to have a meaningful impact it must also allow the player to do a large amount of things which is why there is a destructable/constructable world that the player can interact with using the gun decks.

## Cards

### BaseCard
- Projectile(ProjectileModifier[])
- Multicast(BaseCard[])
- Effect(Effect)

### ProjectileModifier
- Increace damage
- Decrease damage
- Increase speed
- Decrease speed
- Increase size
- Decrease size
- Increase lifetime
- Decrease lifetime
- Increace health
- Add knockback
- Over time(ProjectileModifier)
- Add gravity
- No friendly fire
- Only friendly fire
- Piercing
- Trail(Material)
- On hit(BaseCard)
- No collision

### Effect
- Increace damage recieved
- Decreace damage recieved
- Increace damage dealt
- Decrease damage dealt
- Increace movement speed
- Decreace movement speed
- Increace attack speed
- Decreace attack speed

## Materials

- Rock
- Dirt
- Ice
- Indestructible
- Air
