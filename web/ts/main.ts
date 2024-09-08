import { resolve_authorization_result } from "./common";

// 定义接口类型
interface AuthorizationResponse {
    user_id: string;
    access_token: string;
}
resolve_authorization_result().then((response: AuthorizationResponse | null) => {
    if (response && response.user_id && response.access_token) {
        console.log('Extension is installed:', response);
        localStorage.setItem("user_id", response.user_id);
        localStorage.setItem("access_token", response.access_token);
    } else {
        console.log('Extension not installed or not responding, or received invalid data');
        if (!localStorage.getItem('access_token')) {
            login();  // 如果没有登录信息，调用 login 函数
        }
    }
}).catch(error => {
    console.error('Error during authorization:', error);
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