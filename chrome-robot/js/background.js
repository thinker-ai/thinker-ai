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