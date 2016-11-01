# -*- coding: utf-8 -*-
"""
"""

import os
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
        render_template, flash


# create our little application
app = Flask(__name__)

# Load default config and override config from an environment variable
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'flaskr.db'),
    DEBUG=True,
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
    ))
# 如果有FLASKR_SETTINGS这个环境变量就用环境变量的值
# 否则就用上面update之后的app.config
app.config.from_envvar('FLASKR_SETTINGS', silent=True)


# 不针对某一个context 建立一个总体的接口
def connect_db():
    """Connects to the specific database."""
    rv = sqlite3.connect(app.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv


def init_db():
    """Initializes the database."""
    # 由于应用环境在每次每次请求传入时创建
    # 因此要手动创建一个应用环境 而g在应用环境外无法获知它属于哪个应用
    # 在with语句内部 g对象会与app关联
    # 在离开with函数后
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            # SQLite数据库连接对象提供了一个游标对象
            # 游标上有一个方法可以执行完整的脚本
            db.cursor().executescript(f.read())
        db.commit()


@app.cli.command('initdb')
def initdb_command():
    """Creates the database tables."""
    init_db()
    print('Initialized the database.')

# 针对某个context构建一个数据库连接
# 需要记住 'g' 是一个与当前应用环境有关的通用变量
def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db


# 针对某个context关闭数据库连接 并销毁context
@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


# 一个完整后端读取数据到前端展示的流程
@app.route('/')
def show_entries():
    # stp1. 获得一个数据库连接
    db = get_db()
    # stp2. 执行sql命令
    cur = db.execute('select title, text from entries order by id desc')
    # stp3. 解析从数据库获得的数据
    entries = [dict(title=row[0], text=row[1]) for row in cur.fetchall()]
    # stp4. 在视图函数中渲染模板 返回前端
    return render_template('show_entries.html', entries=entries)

# 一个完整的前端POST数据 后端插入数据库
@app.route('/add', methods=['POST'])
def add_entry():
    if not session.get('logged_in'):
        abort(401)
    db = get_db()
    db.execute('insert into entries (title, text) values (?, ?)',
            [request.form['title'], request.form['text']])
    db.commit()
    flash('New entry was successfully posted')
    return redirect(url_for('show_entries'))


# 目的是在session中增加logged_in属性
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != app.config['USERNAME']:
            error = 'Invalid username'
        elif request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('show_entries'))
    return render_template('login.html', error=error)


# 从session中删除logged_in属性
@app.route('/logout')
def logout():
    # 可以省去检查用户是否登录的属性
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('show_entries'))
