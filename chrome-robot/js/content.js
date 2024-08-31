// 存储数据
const token = localStorage.getItem('access_token');
const user_id = localStorage.getItem('user_id');

if (token && user_id) {
    chrome.runtime.sendMessage({
        action: 'storeData',
        token: token,
        user_id: user_id
    }, (response) => {
        console.log('Response from background:', response);
    });
}

// 读取数据
chrome.runtime.sendMessage({ action: 'getData' }, (response) => {
    if (response) {
        console.log('Retrieved in content script:', response);
    } else {
        console.log('No data retrieved in content script.');
    }
});