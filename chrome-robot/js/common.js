function getToken() {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get('access_token', (result) => { // 使用 'access_token' 作为键
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
            } else {
                resolve(result.access_token); // 获取 'access_token'
            }
        });
    });
}

function loadAxios() {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = chrome.runtime.getURL('js/axios.min.js');
        script.onload = function () {
            // 在 axios 加载完毕后，设置拦截器
            const axiosInstance = axios.create();

            axiosInstance.interceptors.request.use(async function (config) {
                const token = await getToken(); // 等待获取到 token
                if (token) {
                    config.headers.Authorization = 'Bearer ' + token;
                }
                return config;
            }, function (error) {
                return Promise.reject(error);
            });

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

// 将 RequestSender 实例导出
const requestSender = new RequestSender();