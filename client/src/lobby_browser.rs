use tokio::{runtime::Runtime, sync::broadcast::error};

use crate::settings_manager::Settings;
use std::sync::{Arc, Mutex};
use voxel_shared::Lobby;

pub struct LobbyBrowser {
    lobby_list: Arc<Mutex<Vec<Lobby>>>,
    error: Arc<Mutex<Option<String>>>,
    runtime: Runtime,
}

async fn fetch_data(url: String, result: Arc<Mutex<Vec<Lobby>>>, error: Arc<Mutex<Option<String>>>) {
    let request = reqwest::get(format!("http://{}lobby_list", url)).await;
    let request = match request {
        Ok(request) => request,
        Err(e) => {
            let Ok(mut error) = error.lock() else {
                println!("error fetching lobby list: failed to lock error");
                return;
            };
            println!("error fetching lobby list: {:?}", e);
            *error = Some(e.to_string());
            return;
        }
    };
    let lobby_list = request.json::<Vec<Lobby>>().await;
    let lobby_list = match lobby_list {
        Ok(lobby_list) => lobby_list,
        Err(e) => {
            let Ok(mut error) = error.lock() else {
                println!("error fetching lobby list: failed to lock error");
                return;
            };
            println!("error fetching lobby list: {:?}", e);
            *error = Some(e.to_string());
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
            error: Arc::new(Mutex::new(None)),
            runtime: Runtime::new().unwrap(),
        }
    }

    pub fn update(&mut self, settings: &Settings) {
        let future = Box::pin(fetch_data(
            settings.remote_url.clone(),
            self.lobby_list.clone(),
            self.error.clone(),
        ));
        self.runtime = Runtime::new().unwrap();
        self.runtime.spawn(future);
    }

    pub fn get_lobbies(&self) -> Result<Vec<Lobby>, String> {
        if let Ok(mut error) = self.error.lock() {
            if let Some(error_text) = error.clone() {
                *error = None;
                return Err(error_text);
            }
        } else {
            return Err("Could not get lock on error".to_string());
        }
        let Ok(lobby_list) = self.lobby_list.lock() else {
            println!("error fetching lobby list: failed to lock lobby list");
            return Err("Could not get lock on lobby list".to_string());
        };
        Ok(lobby_list.clone())
    }
}
