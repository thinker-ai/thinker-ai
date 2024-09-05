import {RequestSenderBackground} from "./request_sender_background";
import {WebSocketSenderBackgroundWithCallback} from "./web_socket_background";

chrome.runtime.onInstalled.addListener(() => {
  console.log("AI Chat Assistant installed");
});
chrome.action.onClicked.addListener((tab) => {
  chrome.system.display.getInfo((displays) => {
    // 检查 displays 是否定义并且数组长度大于 0
    if (displays && displays.length > 0) {
      const display = displays[0]; // 获取第一个显示器的信息
      const screenWidth = display.bounds.width;

      // 计算右上角的位置
      const windowWidth = 400;
      const windowHeight = 600;
      const top = 0;
      const left = screenWidth - windowWidth;

      chrome.windows.create({
        url: 'popup.html',
        type: 'popup',
        width: windowWidth,
        height: windowHeight,
        top: top,
        left: left,
      });
    } else {
      console.error("无法获取显示器信息：displays 未定义或为空");
    }
  });
});


let request_sender_background:RequestSenderBackground
let web_socket_sender_background:WebSocketSenderBackgroundWithCallback
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    chrome.storage.local.get(['access_token', 'user_id'], (result) => {
        // 使用 'access_token' 作为键
        if (chrome.runtime.lastError) {
            console.error('User ID or access token not found in chrome.storage.local')
        } else {
            request_sender_background = new RequestSenderBackground("libs/axios.min.js", result.access_token, sendResponse);
            web_socket_sender_background = new WebSocketSenderBackgroundWithCallback(sendResponse)
            web_socket_sender_background.connect(result.access_token)
        }
    });
    if (message.action === 'storeData') {
        const dataToStore = message.data || {};  // 从消息中获取要存储的数据对象
        chrome.storage.local.set(dataToStore, function() {
            console.log('Data stored in background script:', dataToStore);
            sendResponse({ status: 'success' });
        });
        return true;  // 表示异步响应
    }

    if (message.action === 'getData') {
        const keys = message.keys || []; // 从消息中获取要检索的键数组
        chrome.storage.local.get(keys, function(items) {
            if (Object.keys(items).length > 0) {
                console.log('Data retrieved in background script:', items);
                sendResponse(items);  // 直接返回获取到的所有数据
            } else {
                console.log('Failed to retrieve data in background script.');
                sendResponse(null);
            }
        });
        return true;  // 表示异步响应
    }

    if (message.action === "getActiveTab") {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          sendResponse(tabs[0]);
        });
        return true; // 表示异步响应
    }

    if (message.action === 'login') {
        login(message.username, message.password, sendResponse);
        return true;  // 表示响应是异步的
    }

    if (message.action === 'getAuthorization') {
        getAuthorization(sendResponse);
        return true;  // 表示响应是异步的
    }

    if (message.action === 'sendMessage') {
        try {
            web_socket_sender_background.sendMessage({
                action: 'sendMessage',
                content: message.content
            });
        } catch (error) {
            console.error('Failed to send message:', error);
        }
    }
    if (message.action === 'ai_action_result') {
        on_ai_action_result(message.data)
    }
})


function login(username:string, password:string, send_response: (response?: any) => void) {
    const params = new URLSearchParams({
        username: username,
        password: password
    });

    // 使用 request_sender 发送请求
    request_sender_background.makeRequest(
        'post',
        'http://0.0.0.0:8000/login',
         params,
        false,
        (data:any)=>{
                            if (data && data.access_token && data.user_id) {
                                const accessToken = data.access_token;
                                const userId = data.user_id;

                                // 将登录结果存储到 chrome.storage.local
                                chrome.storage.local.set({ access_token: accessToken, user_id: userId }, () => {
                                    if (chrome.runtime.lastError) {
                                        send_response({ status: 'error', message: chrome.runtime.lastError.message });
                                    } else {
                                        send_response({ status: 'success' });
                                    }
                                });
                            } else {
                                send_response({ status: 'error', message: 'Invalid login response' });
                            }
                         },
        (error) => {
                            console.error('Error:', error);
                            send_response({ status: 'error', message: error });
                        }
       );
}

function getAuthorization(sendResponse: (response?: any) => void) {
    // 从 chrome.storage.local 中读取 access_token 和 user_id
    chrome.storage.local.get(['access_token', 'user_id'], (items) => {
        if (chrome.runtime.lastError) {
            sendResponse(null);
        } else {
            sendResponse({
                access_token: items.access_token,
                user_id: items.user_id
            });
        }
    });
}



function on_ai_action_result(message:any) {
    web_socket_sender_background.sendMessage(message);
}