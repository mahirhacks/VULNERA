/*
 * test_13.c — post-2019 temporal verification sample
 *
 * Inspired by: CVE-2022-2588 (Linux cls_route UAF, disclosed Aug 2022)
 * CWE: CWE-416 (Use After Free)
 *
 * Expected VULNERA Stage 2: may weakly match CWE-416 neighbors or stay unattributed.
 */

#include <stdlib.h>

typedef struct route_filter route_filter_t;

struct route_filter {
    void (*destroy)(route_filter_t *self);
    void *classifier_state;
};

typedef struct {
    route_filter_t *filter;
    void *packet_ctx;
} classify_job_t;

static void noop_destroy(route_filter_t *self)
{
    (void)self;
}

/*
 * Tear down filter then invoke its destructor callback.
 * Bug: frees filter object then calls destroy through the freed pointer.
 */
int cls_route_teardown(classify_job_t *job)
{
    route_filter_t *filter;
    void (*dtor)(route_filter_t *);

    if (job == NULL || job->filter == NULL) {
        return -1;
    }

    filter = job->filter;
    dtor = filter->destroy;
    free(filter);
    dtor(filter);

    job->filter = NULL;
    return 0;
}

void cls_route_init_job(classify_job_t *job, void *state)
{
    route_filter_t *filter;

    if (job == NULL) {
        return;
    }

    filter = (route_filter_t *)malloc(sizeof(route_filter_t));
    if (filter == NULL) {
        return;
    }

    filter->destroy = noop_destroy;
    filter->classifier_state = state;
    job->filter = filter;
    job->packet_ctx = NULL;
}
