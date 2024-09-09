// 定义接口类型
interface AuthorizationResponse {
    user_id: string;
    access_token: string;
}

function resolve_authorization_result(): Promise<{ user_id: string; access_token: string } | null> {
    return new Promise((resolve, reject) => {
        // 监听来自 content script 的消息
        const messageHandler = (event: MessageEvent) => {
            if (event.data && event.data.action === 'authorizationResult') {
                window.removeEventListener('message', messageHandler);  // 确保只处理一次消息
                resolve(event.data.response);  // 解析消息数据并返回
            }
        };
        window.addEventListener('message', messageHandler);
        // 发送消息给 content script，请求获取 authorizationResult 信息
        window.postMessage({ action: 'getAuthorization' }, '*');
        // 设置超时，如果超过一定时间没有接收到消息，则 reject
        const timeout = setTimeout(() => {
            window.removeEventListener('message', messageHandler);  // 超时后移除事件监听器
            reject('Authorization result not received from extension within timeout period');
        }, 5000); // 5秒超时，可以根据实际需求调整

    });
}
resolve_authorization_result().then((response: AuthorizationResponse | null) => {
    if (response && response.user_id && response.access_token) {
        console.log('Extension is installed:', response);
        localStorage.setItem("user_id", response.user_id);
        localStorage.setItem("access_token", response.access_token);
    } else {
        console.log('Extension is installed but has not login');
    }
}).catch(reason => {
    console.log(reason);
    if (!localStorage.getItem('access_token')) {
        login();  // 如果没有登录信息，调用 login 函数
    }
});

// 定义 login 函数，带有 fetch 请求
function login(): void {
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded', // 修改为表单数据类型
        },
        body: new URLSearchParams({
            username: 'testuser',
            password: 'testpassword',
        }),
    })
    .then(response => response.json())  // 推断 response.json() 返回的是一个对象
    .then((data: { access_token: string; user_id: string }) => {
        const token = data.access_token;
        const user_id = data.user_id;
        localStorage.setItem('access_token', token);
        localStorage.setItem('user_id', user_id);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}