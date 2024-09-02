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
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
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

    if (message.action === 'getSocketState') {
        sendResponse({ state: socket ? socket.readyState : WebSocket.CLOSED });
    }

    if (message.action === 'login') {
        login(message.username, message.password, sendResponse);
        return true;  // 表示响应是异步的
    }

    if (message.action === 'getAuthorization') {
        getAuthorization(sendResponse);
        return true;  // 表示响应是异步的
    }
});

function login(username, password, sendResponse) {
    fetch('http://0.0.0.0:8000/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            username: username,
            password: password,
        }),
    })
    .then(response => {
        if (!response.ok) {  // 检查响应状态码是否是 2xx
            throw new Error(`HTTP error! status: ${response.status}`); // 抛出错误，以便在 .catch() 中捕获
        }
        return response.json();  // 如果状态码是 2xx，则继续解析 JSON
    })
    .then(data => {
        // 将登录结果存储到 chrome.storage.local
        const accessToken = data.access_token;
        const userId = data.user_id;

        chrome.storage.local.set({ access_token: accessToken, user_id: userId }, () => {
            if (chrome.runtime.lastError) {
                sendResponse({ status: 'error', message: chrome.runtime.lastError.message });
            } else {
                sendResponse({ status: 'success' });
            }
        });
    })
    .catch(error => {
        console.error('Error:', error);
        sendResponse({ status: 'error', message: error.message });
    });
}

function getAuthorization(sendResponse) {
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