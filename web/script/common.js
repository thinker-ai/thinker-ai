function showContent(frame_id,menu_class_name,page, element) {
    // Update the iframe's src attribute to load the corresponding page
    document.getElementById(frame_id).src = page;
    // Update the active state of the menu items
    var menuItems = document.getElementsByClassName(menu_class_name);
    for (var i = 0; i < menuItems.length; i++) {
        menuItems[i].classList.remove('active');
    }
    element.classList.add('active');
}

function loadAxios() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js';
        script.onload = function () {
            // 在 axios 加载完毕后，设置拦截器
            const axiosInstance = axios.create();

            axiosInstance.interceptors.request.use(
                function (config) {
                    const token = localStorage.getItem('access_token');
                    if (token) {
                        config.headers.Authorization = 'Bearer ' + token;
                    }
                    return config;
                },
                function (error) {
                    return Promise.reject(error);
                }
            );

            resolve(axiosInstance); // 返回设置了拦截器的 axios 实例
        };
        script.onerror = function () {
            reject(new Error('Failed to load axios'));
        };
        document.head.appendChild(script);
    });
}

function RequestSender() {
    this.callbacks = []; // 用于存储回调函数

    // 注册回调函数
    this.registerCallback = function(callback) {
        this.callbacks.push(callback);
    };

    // 发送请求
    this.makeRequest = function({ method, url, params }, { onError, clientParams, responseParamsExtractor }) {
        loadAxios().then(function(axiosInstance) {
            let request;
            if (method === 'get') {
                request = axiosInstance.get(url, { params });
            } else if (method === 'post') {
                request = axiosInstance.post(url, params, {
                    headers: { 'Content-Type': 'application/json' }
                });
            }

            request.then(function(response) {
                if (response.status === 200) {
                    const responseParams = responseParamsExtractor(response);
                    this.callbacks.forEach(function(callback) {
                        callback(clientParams, responseParams); // 分发响应参数给每个注册的回调函数
                    });
                } else {
                    alert('HTTP error! status: ' + response.status);
                    throw new Error('HTTP error! status: ' + response.status);
                }
            }.bind(this)).catch(function(e) {
                if (onError) {
                    onError(e);
                } else {
                    alert('Error: ' + e.message);
                    console.error('Error:', e);
                }
            });
        }.bind(this)).catch(function(error) {
            alert('Failed to load axios: ' + error.message);
            console.error('Error:', error);
        });
    };
}

const requestSender = new RequestSender();
let reconnectInterval = 1000; // 1 second
let socket;
function connect() {
    const user_id = localStorage.getItem("user_id");
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
                // chrome.runtime.sendMessage({ action: 'openTab', url: url, name: data.name });
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
}
function checkExtension() {
    return new Promise((resolve) => {
        // 监听来自 content script 的消息
        window.addEventListener('message', function(event) {
            if (event.data && event.data.action === 'authorizationResult') {
                resolve(event.data.response);
            }
        });

        // 发送消息给 content script，请求获取授权信息
        window.postMessage({ action: 'getAuthorization' }, '*');
    });
}

function loadWebSocketScriptIfNoPlugin() {
    if (!!window.chrome) {
        checkExtension().then((response) => {
            if (response) {
                console.log('插件已安装，跳过加载 WebSocket 脚本。');
            } else {
                console.log('Extension not installed or not responding');
                connect();
            }
        })
    }else {
        connect();
    }
}


loadWebSocketScriptIfNoPlugin()
