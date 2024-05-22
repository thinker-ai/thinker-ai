function login() {
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
    .then(
        response => response.json()
    ).then(data => {
        const token = data.access_token;
        const user_id = data.user_id;
        localStorage.setItem('access_token', token);
        localStorage.setItem('user_id', user_id);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// 创建 axios 实例
const instance = axios.create();

// 添加请求拦截器
instance.interceptors.request.use(config => {
    const token = localStorage.getItem('access_token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
}, error => {
    return Promise.reject(error);
});