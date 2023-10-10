cargo build --release
mkdir voxel-game
Copy-Item -Path "target/release/voxel-game.exe" -Destination "voxel-game/voxel-game.exe"
Copy-Item -Path "assets" -Destination "voxel-game/assets" -Recurse
Copy-Item -Path "decks" -Destination "voxel-game/decks" -Recurse
Copy-Item -Path "settings.yaml" -Destination "voxel-game/settings.yaml"
Compress-Archive -Path voxel-game -DestinationPath voxel-game.zip
