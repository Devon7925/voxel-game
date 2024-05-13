use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

use crate::{game_manager::GameSettings, networking::RoomId, settings_manager::Settings};
use std::sync::{Arc, Mutex};

pub struct LobbyBrowser {
    lobby_list: Arc<Mutex<Vec<Lobby>>>,
    runtime: Runtime,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Lobby {
    pub name: String,
    pub lobby_id: RoomId,
    pub settings: GameSettings,
}

async fn fetch_data(url: String, result: Arc<Mutex<Vec<Lobby>>>) {
    let request = reqwest::get(format!("http://{}lobby_list", url))
        .await;
    let request = match request {
        Ok(request) => request,
        Err(e) => {
            println!("error fetching lobby list: {:?}", e);
            return;
        }
    };
    let lobby_list = request
        .json::<Vec<Lobby>>()
        .await;
    let lobby_list = match lobby_list {
        Ok(lobby_list) => lobby_list,
        Err(e) => {
            println!("error fetching lobby list: {:?}", e);
            return;
        }
    };
    let Ok(mut result_mut) = result.lock() else {
        println!("error fetching lobby list: failed to lock lobby list");
        return;
    };
    *result_mut = lobby_list;
}

impl LobbyBrowser {
    pub fn new() -> Self {
        LobbyBrowser {
            lobby_list: Arc::new(Mutex::new(Vec::new())),
            runtime: Runtime::new().unwrap(),
        }
    }

    pub fn update(&mut self, settings: &Settings) {
        let future = Box::pin(fetch_data(settings.remote_url.clone(), self.lobby_list.clone()));
        self.runtime = Runtime::new().unwrap();
        self.runtime.spawn(future);
    }

    pub fn get_lobbies(&self) -> Vec<Lobby> {
        let Ok(lobby_list) = self.lobby_list.lock() else {
            println!("error fetching lobby list: failed to lock lobby list");
            return Vec::new();
        };
        lobby_list.clone()
    }
}
