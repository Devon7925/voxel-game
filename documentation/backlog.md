# SOON
* Documentation
    * Update game_design.md
        * Explain reasoning for what cards exist
        * Remove things that weren't added
    * Move issue tracking to github
* Card Editor
    * Add up/down arrows on hover for number cards for more intuitive editing
    * Keybinds
        * Editor
        * Keybind Chords
* Add a way to peek at other player's decks
* Rendering
    * FXAA https://en.wikipedia.org/wiki/Fast_approximate_anti-aliasing
    * Improve water waves to not be directional
    * Depth based shadows
    * Improved player model
* Improve the experience of picking from prebuilt decks
* Cards
    * Make passive type abilities work better
    * Add cooldown modifier that makes all charges reload at once
    * Homing cards so turrets can exist
    * Add energy system
        * Energy buff cards
            * simple modifiers
            * effect durations
            * damage/knockback amounts
            * duplication
            * trail
        * Cooldown modifier to reduce cooldown at energy cost
        * Gain energy card
    * Directions
        * Add directions
            * Up
            * Facing
            * Movement
        * Add as parameter to
            * Knockback
            * Gravity

# EVENTUALLY
* Animations
    * Walking
    * Crouching
* Enemies
* Settings Editor
    * Could be easily done by https://crates.io/crates/egui-probe/
        * Blocked by incompatible versions of egui between egui-probe and egui_winit_vulkano
            * Blocked by https://github.com/hakolao/egui_winit_vulkano/pull/53
                * Blocked by vulkano update
* Card Editor
    * Modifier icons
* Basic sounds
    * Player Step
    * Hit sounds
    * Headshot sounds


