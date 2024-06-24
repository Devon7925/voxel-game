# SOON
* Documentation
    * Update game_design.md
        * Explain reasoning for what cards exist
        * Remove things that weren't added
    * Write readme.md
        * Explain what the game is
        * Add screenshots
        * Client build instructions
        * Server run instructions
* Card Editor
    * Keybinds
        * Editor
        * Keybind Chords
* Rendering
    * FXAA https://en.wikipedia.org/wiki/Fast_approximate_anti-aliasing
    * Improve water waves to not be directional
    * Refraction
    * Improved player model
* Cards
    * Shortcut cards
        * Pinpoint to reduce size to 0
        * Squash to reduce height to 0
        * Squish to reduce width to 0
        * Squelch to reduce length to 0
        * Instant to reduce lifetime to 0
    * No player collision card
    * Merge status effects and projectile modifiers
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
* Movement tweaks
    * Buff wall jump
    * Make diagonal movement faster but not root 2 faster
    * Nerf jump decreased gravity
    * Fix step up bug
* Other
    * Fix DOT hitmarkers

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


