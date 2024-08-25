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
