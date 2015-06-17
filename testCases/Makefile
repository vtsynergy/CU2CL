SUBDIRS = passing failing

.PHONY: subdirs $(SUBDIRS)
subdirs: $(SUBDIRS) utils
$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: all
all: utils subdirs

.PHONY: clean
clean:
	rm *.o
	for dir in $(SUBDIRS); do make -C $$dir clean; done

.PHONY: utils
utils: test_utils.o

test_utils.o: test_utils.cpp
	g++ test_utils.cpp -c
