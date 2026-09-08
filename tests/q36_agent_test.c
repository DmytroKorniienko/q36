#define Q36_AGENT_TEST
#define Q36_AGENT_TEST_NO_MAIN
#include "../q36_agent.c"

/* Behave like sudo's terminal password reader without using real credentials
 * or requiring privilege. stdout/stderr still belong to the bash tool. */
static int password_child(void) {
    int fd = open("/dev/tty", O_RDWR);
    if (fd < 0) { puts("NO_TERMINAL"); return 1; }
    struct termios saved, mode;
    if (tcgetattr(fd, &saved) != 0) return 2;
    mode = saved;
    mode.c_lflag &= ~(ECHO | ECHONL);
    for (int attempt = 0; attempt < 3; attempt++) {
        if (tcsetattr(fd, TCSAFLUSH, &mode) != 0) return 3;
        write_all(fd, "[sudo] password for test: ", 26);
        char buf[1024] = {0};
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        tcsetattr(fd, TCSANOW, &saved);
        if (n > 0 && !strcmp(buf, "q36-test-secret\n")) {
            close(fd);
            puts("PASSWORD_OK");
            return 0;
        }
        write_all(fd, "\nSorry, try again.\n", 19);
    }
    close(fd);
    return 1;
}

static void test_worker_init(agent_worker *w, agent_config *cfg) {
    memset(w, 0, sizeof(*w));
    w->cfg = cfg;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    if (pipe(w->wake_fd) != 0) abort();
    set_nonblock(w->wake_fd[0], true, NULL);
    set_nonblock(w->wake_fd[1], true, NULL);
}

static void test_worker_free(agent_worker *w) {
    agent_bash_jobs_free(w);
    free(w->out);
    close(w->wake_fd[0]);
    close(w->wake_fd[1]);
    pthread_mutex_destroy(&w->mu);
    pthread_cond_destroy(&w->cond);
}

static void *password_job_thread(void *arg) {
    agent_worker *w = arg;
    agent_bash_refresh_for(w, w->bash_jobs, 10);
    agent_set_status(w, AGENT_WORKER_IDLE);
    return NULL;
}

static int password_driver(const char *cmd, int timeout, bool noninteractive) {
    agent_config cfg = {.non_interactive = noninteractive};
    agent_worker w;
    test_worker_init(&w, &cfg);
    agent_editor editor = {0};
    const char *prompt = "q36-agent> ";
    if (editor_start(&editor, prompt, "test", "unfinished draft") != 0) return 1;
    char err[160];
    agent_bash_job *job = agent_bash_start(&w, cmd, timeout, err, sizeof(err));
    if (!job) { fprintf(stderr, "%s\n", err); return 2; }
    w.status.state = AGENT_WORKER_GENERATING;
    pthread_t thread;
    if (pthread_create(&thread, NULL, password_job_thread, &w) != 0) return 3;
    for (;;) {
        struct pollfd pfd = {.fd = w.wake_fd[0], .events = POLLIN};
        poll(&pfd, 1, 100);
        drain_wake_fd(w.wake_fd[0]);
        agent_password_request request;
        if (worker_take_password_request(&w, &request)) {
            if (editor_prompt_password(&editor, &w, &request, prompt, "test") != 0)
                abort();
        }
        pthread_mutex_lock(&w.mu);
        bool done = w.status.state == AGENT_WORKER_IDLE;
        w.wake_pending = false;
        pthread_mutex_unlock(&w.mu);
        if (done) break;
    }
    pthread_join(thread, NULL);
    bool draft_ok = !strcmp(editor.edit.buf, "unfinished draft");
    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    char *obs = agent_bash_observation(job, false);
    printf("\nDRAFT_OK=%d\n%s", draft_ok, obs);
    free(obs);
    unlink(job->path);
    test_worker_free(&w);
    return draft_ok ? 0 : 4;
}

static void test_bash_noninteractive(void) {
    agent_config cfg = {.non_interactive = true};
    agent_worker w;
    test_worker_init(&w, &cfg);
    char err[160];
    agent_bash_job *job = agent_bash_start(&w,
        "printf stdout; printf stderr >&2; read value; test $? -ne 0", 5,
        err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (job) {
        agent_bash_refresh_for(&w, job, 5);
        AGENT_TEST_ASSERT(!job->running && job->exit_status == 0);
        char *obs = agent_bash_observation(job, false);
        AGENT_TEST_ASSERT(strstr(obs, "stdoutstderr") != NULL);
        free(obs);
        unlink(job->path);
    }
    test_worker_free(&w);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--password-child")) return password_child();
    if (argc == 4 && !strcmp(argv[1], "--password-driver"))
        return password_driver(argv[2], atoi(argv[3]), false);
    if (argc == 4 && !strcmp(argv[1], "--noninteractive-driver"))
        return password_driver(argv[2], atoi(argv[3]), true);
    q36_agent_unit_tests_run();
    test_bash_noninteractive();
    if (agent_test_failures) {
        fprintf(stderr, "q36-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("q36-agent tests: ok");
    return 0;
}
