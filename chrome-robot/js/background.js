chrome.runtime.onInstalled.addListener(() => {
  console.log("AI Chat Assistant installed");
});

chrome.action.onClicked.addListener((tab) => {
  chrome.system.display.getInfo((displays) => {
    // 检查 displays 是否定义并且数组长度大于 0
    if (displays && displays.length > 0) {
      const display = displays[0]; // 获取第一个显示器的信息
      const screenWidth = display.bounds.width;
      const screenHeight = display.bounds.height;

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
        chrome.storage.local.set({ 'access_token': message.token, 'user_id': message.user_id }, function() {
            console.log('Data stored in background script.');
            sendResponse({ status: 'success' });
        });
        return true;  // 表示异步响应
    }

    if (message.action === 'getData') {
        chrome.storage.local.get(['access_token', 'user_id'], function(items) {
            if (items.access_token && items.user_id) {
                console.log('Data retrieved in background script.');
                sendResponse({
                    access_token: items.access_token,
                    user_id: items.user_id
                });
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
});


let reconnectInterval = 1000; // 1 second
let socket;

function connect() {
    chrome.storage.local.get('user_id', (result) => {
        const user_id = result.user_id;
        if (user_id) {
            // 如果已有连接未断开，不再建立新连接
            if (socket && socket.readyState !== WebSocket.CLOSED && socket.readyState !== WebSocket.CLOSING) {
                console.log('A connection is already open. No need to reconnect.');
                return;
            }

            socket = new WebSocket(`ws://localhost:8000/ws/${user_id}`);
            let heartbeatInterval;

            socket.onopen = () => {
                console.log('Connected to server');
                reconnectInterval = 1000; // Reset the interval on successful connection

                // Start sending heartbeat messages every 10 seconds
                heartbeatInterval = setInterval(() => {
                    if (socket.readyState === WebSocket.OPEN) {
                        socket.send(JSON.stringify({ type: 'heartbeat' }));
                    }
                }, 10000);
            };

            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type !== 'heartbeat') {
                    const url = `http://localhost:${data.port}${data.mount_path}`;
                    // Example: send this data to a popup or content script
                    chrome.runtime.sendMessage({ action: 'openTab', url: url, name: data.name });
                }
            };

            socket.onclose = (event) => {
                console.log('Connection closed', event);
                clearInterval(heartbeatInterval); // Stop heartbeat messages

                // 判断是否是由于网络问题或服务器问题引起的关闭
                if (event.wasClean === false) {
                    console.log('Connection closed due to network or server issues.');
                    setTimeout(connect, reconnectInterval); // Attempt to reconnect after a delay
                    // Increment the interval for each failed attempt
                    reconnectInterval = Math.min(reconnectInterval * 2, 5000); // Max 5 seconds
                }
            };

            socket.onerror = (error) => {
                console.log('WebSocket error', error);
                // 处理网络问题或服务器问题引发的错误
                socket.close(); // 关闭当前连接，触发 onclose 事件，进而重新连接
            };
        }
    });
}

// Call the connect function to start the WebSocket connection
connect();

// Handle incoming messages from other parts of the extension
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'getSocketState') {
        sendResponse({ state: socket ? socket.readyState : WebSocket.CLOSED });
    }
});