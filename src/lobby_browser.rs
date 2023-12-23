use serde::{Deserialize, Serialize};

use crate::{settings_manager::Settings, game_manager::GameSettings};
use std::sync::{Arc, Mutex};

pub struct LobbyBrowser {
    lobby_list: Arc<Mutex<Vec<Lobby>>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Lobby {
    pub name: String,
    pub lobby_id: u64,
    pub settings: GameSettings,
}

async fn fetch_data(url: String) -> Result<Vec<Lobby>, reqwest::Error> {
    reqwest::get(format!("http://{}lobby_list", url))
        .await?
        .json::<Vec<Lobby>>()
        .await
}

impl LobbyBrowser {
    pub fn new() -> Self {
        LobbyBrowser {
            lobby_list: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn update(&mut self, settings: &Settings) {
        let lobby_list_update = reqwest::blocking::get("http://primd.net:3536/lobby_list");
        match lobby_list_update {
            Ok(lobby_list) => {
                let Ok(mut lobby_list_mut) = self.lobby_list.lock() else {
                    println!("error fetching lobby list: failed to lock lobby list");
                    return;
                };
                *lobby_list_mut = lobby_list.json::<Vec<Lobby>>().unwrap();
            }
            Err(e) => {
                println!("error fetching lobby list: {:?}", e);
            }
        }
        // let lobby_list_clone = self.lobby_list.clone();

        // let rt = Runtime::new().unwrap();
        // let url = settings.remote_url.clone();
        // let future = fetch_data(url).fuse();
        // rt.spawn(async move {
        //     let timeout = Delay::new(Duration::from_millis(50));
        //     futures::pin_mut!(future, timeout);
        //     loop {
        //         select! {
        //             _ = (&mut timeout).fuse() => {
        //                 timeout.reset(Duration::from_millis(50));
        //             }

        //             x = &mut future => {
        //                 match x {
        //                     Ok(lobby_list) => {
        //                         let Ok(mut lobby_list_mut) = (*lobby_list_clone).lock() else {
        //                             println!("error fetching lobby list: failed to lock lobby list");
        //                             break;
        //                         };
        //                         *lobby_list_mut = lobby_list.clone();
        //                     }
        //                     Err(e) => {
        //                         println!("error fetching lobby list: {:?}", e);
        //                     }
        //                 }
        //                 break;
        //             }
        //         }
        //     }
        // });
    }

    pub fn get_lobbies(&self) -> Vec<Lobby> {
        let Ok(lobby_list) = self.lobby_list.lock() else {
            println!("error fetching lobby list: failed to lock lobby list");
            return Vec::new();
        };
        lobby_list.clone()
    }
}