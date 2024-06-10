# SOON
* Documentation
    * Update game_design.md
        * Explain reasoning for what cards exist
        * Remove things that weren't added
* Card Editor
    * Keybinds
        * Editor
        * Keybind Chords
* Rendering
    * FXAA https://en.wikipedia.org/wiki/Fast_approximate_anti-aliasing
    * Improve water waves to not be directional
    * Depth based shadows
    * Improved player model
* Cards
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
    * Hurt sounds


