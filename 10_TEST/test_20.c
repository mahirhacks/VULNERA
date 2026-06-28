/*
 * test_20.c
 *
 * Inspired by: CVE-2023-32250 (Linux ksmbd session race UAF, May 2023)
 * CWE: CWE-362 (Race Condition) -> CWE-416 (Use After Free)
 *
 * Bug: session lookup and session destroy are not serialised. A concurrent
 * disconnect frees the session object while the handler still holds a
 * pointer to it and later dereferences ->state.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    int   session_id;
    int   state;
    char  username[64];
    void *auth_token;
} smb_session_t;

typedef struct {
    smb_session_t **sessions;
    int             count;
    int             capacity;
} smb_server_t;

smb_session_t *smb_session_lookup(smb_server_t *srv, int session_id)
{
    int i;
    if (srv == NULL || srv->sessions == NULL)
        return NULL;

    for (i = 0; i < srv->count; i++) {
        if (srv->sessions[i] != NULL &&
            srv->sessions[i]->session_id == session_id)
            return srv->sessions[i];
    }
    return NULL;
}

void smb_session_destroy(smb_server_t *srv, int session_id)
{
    int i;
    if (srv == NULL)
        return;

    for (i = 0; i < srv->count; i++) {
        if (srv->sessions[i] != NULL &&
            srv->sessions[i]->session_id == session_id) {
            free(srv->sessions[i]->auth_token);
            free(srv->sessions[i]);
            srv->sessions[i] = NULL;
            return;
        }
    }
}

/*
 * BUG: no lock between lookup and use. Thread A calls smb_handle_request
 * and holds 'sess'. Thread B calls smb_session_destroy on the same id
 * between the lookup and the sess->state read — classic TOCTOU UAF.
 */
int smb_handle_request(smb_server_t *srv, int session_id,
                       const char *payload)
{
    smb_session_t *sess;

    if (srv == NULL || payload == NULL)
        return -1;

    sess = smb_session_lookup(srv, session_id);
    if (sess == NULL)
        return -1;

    /* --- gap: concurrent smb_session_destroy(srv, session_id) frees sess --- */

    if (sess->state != 1)      /* UAF read */
        return -1;

    /* process payload using freed session context */
    strncpy(sess->username, payload, sizeof(sess->username) - 1);  /* UAF write */
    return 0;
}
