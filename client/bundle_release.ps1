cargo build --release
mkdir voxel-game
Copy-Item -Path "target/release/voxel-game.exe" -Destination "voxel-game"
Copy-Item -Path "assets" -Destination "voxel-game" -Recurse
Copy-Item -Path "decks" -Destination "voxel-game" -Recurse
Copy-Item -Path "settings.yaml" -Destination "voxel-game"
Compress-Archive -Force -Path voxel-game -DestinationPath voxel-game.zip
Remove-Item -Path voxel-game -Recurse
