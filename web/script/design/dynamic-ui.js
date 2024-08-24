const user_id = localStorage.getItem("user_id");
let reconnectInterval = 1000; // 1 second

function connect() {
    if (user_id) {
        const socket = new WebSocket(`ws://localhost:8000/ws/${user_id}`);
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
                addTabWithUrl(url, data.name);
            }
        };

        socket.onclose = (event) => {
            console.log('Connection closed', event);
            clearInterval(heartbeatInterval); // Stop heartbeat messages
            // Attempt to reconnect after a delay
            setTimeout(connect, reconnectInterval);
            // Increment the interval for each failed attempt
            reconnectInterval = Math.min(reconnectInterval * 2, 5000); // Max 5 seconds
        };

        socket.onerror = (error) => {
            console.log('WebSocket error', error);
        };
    }
}
connect();


// 添加请求拦截器
axios.interceptors.request.use(config => {
    const token = localStorage.getItem('access_token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
}, error => {
    return Promise.reject(error);
});
if(!localStorage.getItem('access_token'))
    login();



function openTab(event, tabId) {
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');

        tabs.forEach(tab => {
            tab.classList.remove('active');
        });

        tabContents.forEach(content => {
            content.classList.remove('active');
        });
        if(tabId) {
            document.getElementById(tabId + '-content').classList.add('active');
        }
        if(event) {
            event.currentTarget.classList.add('active');
        }
    }

let tabCount = 0;
function addTab() {
    const url = prompt("请输入网页地址:");
    if (url) {
        addTabWithUrl(url, `Tab ${tabCount + 1}`);
    }
}

function addTabWithUrl(url, title) {
    tabCount++;
    const newTab = document.createElement('div');
    newTab.className = 'tab';
    newTab.id = 'tab' + tabCount;
    newTab.setAttribute('onclick', `openTab(event, 'tab${tabCount}')`);

    const closeButton = document.createElement('span');
    closeButton.textContent = 'X';
    closeButton.className = 'close-tab';
    closeButton.setAttribute('onclick', `closeTab(event, 'tab${tabCount}')`);

    newTab.textContent = title;
    newTab.appendChild(closeButton);
    document.getElementById('tab-container').appendChild(newTab);

    const newTabContent = document.createElement('div');
    newTabContent.className = 'tab-design';
    newTabContent.id = 'tab' + tabCount + '-design';

    const newIframe = document.createElement('iframe');
    newIframe.className = 'tab-frame';
    newIframe.src = url;
    newTabContent.appendChild(newIframe);

    document.getElementById('design').appendChild(newTabContent);
    // Switch to the new tab
    openTab(null, 'tab' + tabCount);
    saveTabsToLocalStorage();
}
function closeTab(event, tabId) {
    event.stopPropagation();
    const confirmClose = confirm("确定要关闭这个标签页吗？");
    if (confirmClose) {
        const tab = document.getElementById(tabId);
        const tabContent = document.getElementById(tabId + '-content');
        tab.remove();
        tabContent.remove();
        saveTabsToLocalStorage();
    }
}
