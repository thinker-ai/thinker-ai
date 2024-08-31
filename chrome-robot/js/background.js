chrome.runtime.onInstalled.addListener(() => {
  console.log("AI Chat Assistant installed");
});

chrome.action.onClicked.addListener((tab) => {
  chrome.windows.create({
    url: 'popup.html',
    type: 'popup',
    width: 400,
    height: 600
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